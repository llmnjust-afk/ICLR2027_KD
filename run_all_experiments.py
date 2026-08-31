#!/usr/bin/env python3
"""
AGS-DD Full Experiment Runner
Efficiently runs all experiments by loading models once and caching clusters.

GPU 0: ImageNette experiments (AGS, baseline, ablation)
GPU 1: ImageWoof experiments (AGS, baseline, ablation)

After completion, auto-uploads results to GitHub.
"""

import os
import sys
import time
import json
import pickle
import argparse
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from tsne_plots import get_loader, get_features_per_class
from ags import (
    ClassComplexityAnalyzer,
    AdaptiveStopTiming,
    TimestepAdaptiveSchedule,
    AGSSampler,
)

LOG_DIR = "./experiment_logs"
RESULTS_DIR = "./results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg, log_file):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def load_model_and_vae(device, image_size=256, vae_type="mse"):
    """Load DiT model and VAE on the specified device."""
    latent_size = image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=latent_size, num_classes=1000
    ).to(device)
    
    ckpt_path = f"DiT-XL-2-{image_size}x{image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    diffusion = create_diffusion("50")
    
    return model, vae, diffusion, latent_size


def compute_clusters_for_dataset(spec, imagenet_dir, vae, device, nclass, 
                                  cags_kmin=2, cags_kmax=20, log_file=None):
    """Compute VAE features and class complexity for a dataset."""
    import argparse as ap
    args_eval = ap.Namespace(
        dataset="imagenet",
        net_type="resnet_ap",
        depth=10, width=1.0, norm_type="instance", nch=3,
        data_path=os.path.join(imagenet_dir, "train"),
        size=224, aug_type="color_crop_cutout", augment=True,
        dseed=0, global_batch_size=1, use_vae=True, finetune_ipc=-1,
        nclass=nclass, spec=spec, phase=0, image_size=256,
    )
    
    if log_file:
        log(f"  Loading images and extracting VAE features...", log_file)
    
    loader = get_loader(args_eval, return_path=True)
    features_per_class, paths_per_class = get_features_per_class(args_eval, loader, vae)
    
    if log_file:
        log(f"  Computing class complexity (CAGS)...", log_file)
    
    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(cags_kmin, cags_kmax),
        alpha=0.5, beta=0.5,
        use_pca=True,
    )
    analyzer.analyze_all_classes(features_per_class, paths_per_class)
    
    if log_file:
        for c in analyzer.complexity_scores:
            log(f"  Class {c}: complexity={analyzer.complexity_scores[c]:.4f}, "
                f"modes={analyzer.mode_counts[c]}", log_file)
    
    return analyzer


def run_single_experiment(
    model, vae, diffusion, latent_size, device,
    analyzer, class_labels, sel_classes, clusters_centers,
    spec, ipc, num_datasets, save_dir, log_file,
    use_cags=True, use_iast=True, use_tags=True,
    tags_schedule="cosine",
    guidance_scale_range=(0.05, 0.3),
    iast_lambda=0.316,
    default_guidance_scale=0.1,
    default_stop_t=0,
    exp_name="ags_full",
):
    """Run a single experiment with specified AGS configuration."""
    
    log(f"  [{exp_name}] Starting: IPC={ipc}, datasets={num_datasets}", log_file)
    
    adaptive_stop = AdaptiveStopTiming(
        t_max=50, lam=iast_lambda, min_stop=5, max_stop_ratio=0.9,
        use_complexity=True,
    )
    guidance_schedule = TimestepAdaptiveSchedule(
        w_max=guidance_scale_range[1],
        schedule_type=tags_schedule,
        warmup_steps=0, decay_rate=3.0,
    )
    
    sampler = AGSSampler(
        model=model, vae=vae, diffusion=diffusion,
        complexity_analyzer=analyzer,
        adaptive_stop=adaptive_stop,
        guidance_schedule=guidance_schedule,
        device=device, num_sampling_steps=50,
        cfg_scale=4.0, latent_size=latent_size,
        use_cags=use_cags, use_iast=use_iast, use_tags=use_tags,
        guidance_scale_range=guidance_scale_range,
    )
    sampler.default_guidance_scale = default_guidance_scale
    sampler.default_stop_t = default_stop_t
    
    args = argparse.Namespace(
        num_samples=ipc,
        num_datasets=num_datasets,
    )
    
    # Compute IPC-specific cluster centers (CAGS uses optimal K for complexity,
    # but generation needs exactly IPC mode centers per class)
    ipc_clusters = analyzer.compute_clusters_for_ipc(ipc, use_pca=True, closest_point=True)
    
    start_time = time.time()
    sampler.generate_dataset(
        args=args,
        class_labels=class_labels,
        sel_classes=sel_classes,
        clusters_centers=ipc_clusters,
        save_dir=save_dir,
        num_datasets=num_datasets,
        use_same_noise=False,
        total_shift=0,
    )
    elapsed = time.time() - start_time
    
    # Save config
    config = sampler.get_config_summary()
    config["exp_name"] = exp_name
    config["spec"] = spec
    config["ipc"] = ipc
    config["num_datasets"] = num_datasets
    config["total_time"] = elapsed
    config["use_cags"] = use_cags
    config["use_iast"] = use_iast
    config["use_tags"] = use_tags
    
    config_path = os.path.join(save_dir, "ags_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    
    # Save as JSON too for easy git tracking
    json_config = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, tuple))}
    json_path = os.path.join(save_dir, "experiment_config.json")
    with open(json_path, "w") as f:
        json.dump(json_config, f, indent=2)
    
    log(f"  [{exp_name}] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)", log_file)
    return elapsed


def setup_classes(spec, nclass):
    """Load class labels for the specified dataset."""
    with open("./misc/class_indices.txt", "r") as fp:
        all_classes = [c.strip() for c in fp.readlines()]
    
    file_map = {
        "woof": "./misc/class_woof.txt",
        "nette": "./misc/class_nette.txt",
        "imagenet100": "./misc/class100.txt",
        "imagenet1k": "./misc/class_indices.txt",
    }
    with open(file_map[spec], "r") as fp:
        sel_classes = [c.strip() for c in fp.readlines()]
    
    sel_classes = sel_classes[:nclass]
    class_labels = [all_classes.index(c) for c in sel_classes]
    return class_labels, sel_classes


def gpu_worker(gpu_id, spec, imagenet_dir, log_file, github_token):
    """Main worker function for a single GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log(f"=== GPU {gpu_id} starting: {spec} ===", log_file)
    
    # Load model once
    log(f"  Loading DiT model and VAE on {device}...", log_file)
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    log(f"  Model loaded.", log_file)
    
    # Setup classes
    class_labels, sel_classes = setup_classes(spec, nclass=10)
    log(f"  Classes: {sel_classes}", log_file)
    
    # Compute clusters once
    log(f"  Computing clusters for {spec}...", log_file)
    analyzer = compute_clusters_for_dataset(
        spec, imagenet_dir, vae, device, nclass=10, log_file=log_file
    )
    clusters_centers = analyzer.cluster_centers
    
    # Save cluster cache
    cluster_cache_path = os.path.join(RESULTS_DIR, f"{spec}_cluster_cache.pkl")
    with open(cluster_cache_path, "wb") as f:
        pickle.dump(analyzer, f)
    log(f"  Cluster cache saved to {cluster_cache_path}", log_file)
    
    experiments = [
        # (name, ipc, num_datasets, use_cags, use_iast, use_tags, tags_schedule,
        #  default_guidance_scale, default_stop_t)
        # Main experiments
        ("ags_full_ipc10", 10, 3, True, True, True, "cosine", 0.1, 25),
        ("ags_full_ipc50", 50, 3, True, True, True, "cosine", 0.1, 25),
        # Baseline
        ("baseline_ipc10", 10, 3, False, False, False, "cosine", 0.1, 0),
        ("baseline_ipc50", 50, 3, False, False, False, "cosine", 0.1, 0),
        # Ablation: individual modules
        ("cags_only_ipc10", 10, 1, True, False, False, "cosine", 0.1, 25),
        ("iast_only_ipc10", 10, 1, False, True, False, "cosine", 0.1, 25),
        ("tags_only_ipc10", 10, 1, False, False, True, "cosine", 0.1, 25),
        # Ablation: pairs
        ("cags_iast_ipc10", 10, 1, True, True, False, "cosine", 0.1, 25),
        ("cags_tags_ipc10", 10, 1, True, False, True, "cosine", 0.1, 25),
        ("iast_tags_ipc10", 10, 1, False, True, True, "cosine", 0.1, 25),
        # Schedule comparison
        ("tags_linear_ipc10", 10, 1, True, True, True, "linear", 0.1, 25),
        ("tags_exp_ipc10", 10, 1, True, True, True, "exponential", 0.1, 25),
    ]
    
    all_results = {}
    for exp_name, ipc, num_ds, cags, iast, tags, sched, def_gs, def_st in experiments:
        save_dir = os.path.join(RESULTS_DIR, f"{spec}_{exp_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            elapsed = run_single_experiment(
                model, vae, diffusion, latent_size, device,
                analyzer, class_labels, sel_classes, clusters_centers,
                spec, ipc, num_ds, save_dir, log_file,
                use_cags=cags, use_iast=iast, use_tags=tags,
                tags_schedule=sched,
                guidance_scale_range=(0.05, 0.3),
                iast_lambda=0.316,
                default_guidance_scale=def_gs,
                default_stop_t=def_st,
                exp_name=exp_name,
            )
            all_results[exp_name] = {
                "exit_code": 0, "time": elapsed, "save_dir": save_dir,
                "ipc": ipc, "num_datasets": num_ds,
            }
        except Exception as e:
            log(f"  [{exp_name}] ERROR: {e}", log_file)
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {"exit_code": 1, "error": str(e)}
    
    # Save results summary
    summary_path = os.path.join(RESULTS_DIR, f"{spec}_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "spec": spec,
            "timestamp": datetime.now().isoformat(),
            "experiments": all_results,
        }, f, indent=2)
    
    log(f"=== GPU {gpu_id} ({spec}) all experiments done ===", log_file)
    
    # Free GPU memory
    del model, vae
    torch.cuda.empty_cache()
    
    return all_results


def upload_to_github(token, log_file):
    """Upload experiment results to GitHub."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"experiment: AGS-DD full results {timestamp}"
    
    log(f"Uploading results to GitHub...", log_file)
    
    # Add results and logs
    subprocess.run(["git", "add", "results/", "experiment_logs/"],
                   cwd=".", capture_output=True)
    
    # Also add the fixed source files
    subprocess.run(["git", "add", "ags/", "sample_ags.py", "run_all_experiments.py",
                    ".gitignore"],
                   cwd=".", capture_output=True)
    
    result = subprocess.run(["git", "commit", "-m", msg],
                           cwd=".", capture_output=True, text=True)
    log(f"Commit: {result.stdout.strip()}", log_file)
    
    push_cmd = [
        "git", "push",
        f"https://{token}@github.com/llmnjust-afk/ICLR2027_KD.git",
        "main"
    ]
    result = subprocess.run(push_cmd, cwd=".", capture_output=True, text=True)
    log(f"Push: {result.stdout.strip()} {result.stderr.strip()}", log_file)
    log(f"Push exit code: {result.returncode}", log_file)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-token", type=str, required=True,
                        help="GitHub PAT for uploading results")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Run only on specified GPU (0 or 1). If None, runs both.")
    parser.add_argument("--spec", type=str, default=None,
                        help="Dataset spec (overrides default per-GPU assignment)")
    parser.add_argument("--imagenet-dir", type=str, default=None,
                        help="Override imagenet directory")
    args = parser.parse_args()
    
    master_log = os.path.join(LOG_DIR, "master.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log(f"{'#'*70}", master_log)
    log(f"# AGS-DD Full Experiment Run - {timestamp}", master_log)
    
    if args.gpu is not None:
        # Single GPU mode
        spec = args.spec or ("nette" if args.gpu == 0 else "woof")
        imagenet_dir = args.imagenet_dir or (
            "/ssd_data/imagenette/imagenette2/" if spec == "nette"
            else "/ssd_data/imagenette/imagewoof2/"
        )
        gpu_log = os.path.join(LOG_DIR, f"gpu{args.gpu}_{spec}.log")
        
        log(f"# GPU {args.gpu}: {spec} (single-GPU mode)", master_log)
        log(f"{'#'*70}", master_log)
        
        results = gpu_worker(args.gpu, spec, imagenet_dir, gpu_log, args.github_token)
        
        # Save results
        summary_path = os.path.join(RESULTS_DIR, f"{spec}_results_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "spec": spec,
                "timestamp": datetime.now().isoformat(),
                "experiments": results,
            }, f, indent=2)
        
        log(f"\nGPU {args.gpu} ({spec}) experiments completed!", master_log)
        
        # Upload to GitHub
        log(f"\nUploading to GitHub...", master_log)
        ret = upload_to_github(args.github_token, master_log)
        
        if ret == 0:
            log(f"GitHub upload successful!", master_log)
        else:
            log(f"GitHub upload failed (exit {ret})", master_log)
    else:
        # Dual GPU mode (uses subprocess internally to avoid device conflicts)
        log(f"# Starting dual-GPU mode via subprocesses", master_log)
        log(f"{'#'*70}", master_log)
        
        import subprocess as sp
        
        procs = []
        for gpu_id, spec, img_dir in [
            (0, "nette", "/ssd_data/imagenette/imagenette2/"),
            (1, "woof", "/ssd_data/imagenette/imagewoof2/"),
        ]:
            cmd = [
                "python3", "run_all_experiments.py",
                "--github-token", args.github_token,
                "--gpu", str(gpu_id),
                "--spec", spec,
                "--imagenet-dir", img_dir,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = sp.Popen(cmd, env=env, stdout=open(
                os.path.join(LOG_DIR, f"gpu{gpu_id}_{spec}.log"), "w"
            ), stderr=sp.STDOUT)
            procs.append((gpu_id, spec, p))
            log(f"Started GPU {gpu_id}: {spec} (PID {p.pid})", master_log)
        
        # Wait for both
        for gpu_id, spec, p in procs:
            p.wait()
            log(f"GPU {gpu_id} ({spec}) finished with exit code {p.returncode}", master_log)
        
        # Upload combined results
        log(f"\nAll experiments completed!", master_log)
        log(f"\nUploading to GitHub...", master_log)
        ret = upload_to_github(args.github_token, master_log)
        
        if ret == 0:
            log(f"GitHub upload successful!", master_log)
        else:
            log(f"GitHub upload failed (exit {ret})", master_log)
    
    log(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", master_log)


if __name__ == "__main__":
    main()
