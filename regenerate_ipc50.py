#!/usr/bin/env python3
"""
Regenerate AGS-DD IPC=50 images with fixed IAST (sqrt scaling, lam=0.316).

Only regenerates the ags_full_ipc50 experiment for both nette and woof.
All other experiments (IPC=10, baseline, ablation) remain unchanged.

Usage:
  python3 regenerate_ipc50.py --github-token <TOKEN>
  python3 regenerate_ipc50.py --github-token <TOKEN> --gpu 0 --spec nette
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime

import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from run_all_experiments import (
    load_model_and_vae,
    compute_clusters_for_dataset,
    setup_classes,
    run_single_experiment,
    upload_to_github,
    log,
    LOG_DIR,
    RESULTS_DIR,
)

IAST_LAMBDA_NEW = 0.316  # sqrt scaling lambda (preserves t_stop at IPC=10)


def gpu_worker_regenerate(gpu_id, spec, imagenet_dir, log_file, github_token=None):
    """Regenerate only ags_full_ipc50 with fixed IAST."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"=== Regenerating IPC=50 on GPU {gpu_id}: {spec} ===", log_file)

    # Load model once
    log(f"  Loading DiT model and VAE on {device}...", log_file)
    model, vae, diffusion, latent_size = load_model_and_vae(device)
    log(f"  Model loaded.", log_file)

    # Setup classes
    class_labels, sel_classes = setup_classes(spec, nclass=10)
    log(f"  Classes: {sel_classes}", log_file)

    # Compute clusters
    log(f"  Computing clusters for {spec}...", log_file)
    analyzer = compute_clusters_for_dataset(
        spec, imagenet_dir, vae, device, nclass=10, log_file=log_file
    )
    clusters_centers = analyzer.cluster_centers

    # Only run ags_full_ipc50 with fixed IAST lambda
    exp_name = "ags_full_ipc50"
    ipc = 50
    num_ds = 3
    save_dir = os.path.join(RESULTS_DIR, f"{spec}_{exp_name}")

    # Backup old images
    import shutil
    backup_dir = os.path.join(RESULTS_DIR, f"{spec}_{exp_name}_old_iast")
    if os.path.exists(save_dir) and not os.path.exists(backup_dir):
        shutil.copytree(save_dir, backup_dir)
        log(f"  Backed up old images to {backup_dir}", log_file)

    # Remove old images
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    log(f"  Regenerating {exp_name} with IAST lambda={IAST_LAMBDA_NEW} (sqrt scaling)...", log_file)

    elapsed = run_single_experiment(
        model, vae, diffusion, latent_size, device,
        analyzer, class_labels, sel_classes, clusters_centers,
        spec, ipc, num_ds, save_dir, log_file,
        use_cags=True, use_iast=True, use_tags=True,
        tags_schedule="cosine",
        guidance_scale_range=(0.05, 0.3),
        iast_lambda=IAST_LAMBDA_NEW,
        default_guidance_scale=0.1,
        default_stop_t=25,
        exp_name=f"{exp_name}_fixed_iast",
    )

    log(f"  {exp_name} regenerated in {elapsed:.1f}s", log_file)

    # Save regeneration summary
    summary = {
        "spec": spec,
        "experiment": exp_name,
        "iast_lambda": IAST_LAMBDA_NEW,
        "iast_scaling": "sqrt",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "save_dir": save_dir,
    }
    summary_path = os.path.join(RESULTS_DIR, f"{spec}_regen_ipc50_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nGPU {gpu_id} ({spec}) IPC=50 regeneration completed!", log_file)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Regenerate AGS-DD IPC=50 images with fixed IAST")
    parser.add_argument("--github-token", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None,
                        help="Run only on specified GPU (0 or 1). If None, runs both.")
    parser.add_argument("--spec", type=str, default=None,
                        help="Dataset spec (overrides default per-GPU assignment)")
    parser.add_argument("--imagenet-dir", type=str, default=None)
    args = parser.parse_args()

    master_log = os.path.join(LOG_DIR, "regen_ipc50_master.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log(f"{'#' * 70}", master_log)
    log(f"# IPC=50 Regeneration with Fixed IAST (sqrt scaling) - {timestamp}", master_log)
    log(f"{'#' * 70}", master_log)

    if args.gpu is not None:
        spec = args.spec or ("nette" if args.gpu == 0 else "woof")
        imagenet_dir = args.imagenet_dir or (
            "/ssd_data/imagenette/imagenette2/" if spec == "nette"
            else "/ssd_data/imagenette/imagewoof2/"
        )
        gpu_log = os.path.join(LOG_DIR, f"regen_ipc50_gpu{args.gpu}_{spec}.log")

        log(f"# GPU {args.gpu}: {spec}", master_log)
        gpu_worker_regenerate(args.gpu, spec, imagenet_dir, gpu_log, args.github_token)
    else:
        log(f"# Starting dual-GPU mode via subprocesses", master_log)

        procs = []
        for gpu_id, spec, img_dir in [
            (0, "nette", "/ssd_data/imagenette/imagenette2/"),
            (1, "woof", "/ssd_data/imagenette/imagewoof2/"),
        ]:
            cmd = [
                "python3", "regenerate_ipc50.py",
                "--gpu", str(gpu_id),
                "--spec", spec,
                "--imagenet-dir", img_dir,
            ]
            if args.github_token:
                cmd.extend(["--github-token", args.github_token])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_log = os.path.join(LOG_DIR, f"regen_ipc50_gpu{gpu_id}_{spec}.log")
            p = subprocess.Popen(
                cmd, env=env,
                stdout=open(gpu_log, "w"),
                stderr=subprocess.STDOUT,
            )
            procs.append((gpu_id, spec, p))
            log(f"Started GPU {gpu_id}: {spec} (PID {p.pid})", master_log)

        for gpu_id, spec, p in procs:
            p.wait()
            log(f"GPU {gpu_id} ({spec}) finished with exit code {p.returncode}", master_log)

        # Upload to GitHub
        if args.github_token:
            log(f"\nUploading to GitHub...", master_log)
            ret = upload_to_github(args.github_token, master_log)
            if ret == 0:
                log(f"GitHub upload successful!", master_log)
            else:
                log(f"GitHub upload failed (exit {ret})", master_log)

    log(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", master_log)


if __name__ == "__main__":
    main()
