#!/usr/bin/env python3
"""
AGS-DD Experiment Runner
Runs experiments on two GPUs in parallel and auto-uploads results to GitHub.
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

LOG_DIR = "./experiment_logs"
RESULTS_DIR = "./results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_command(cmd, log_file, gpu_id=None, env=None):
    """Run a command and log output."""
    full_env = os.environ.copy()
    if gpu_id is not None:
        full_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if env:
        full_env.update(env)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{timestamp}] Running: {' '.join(cmd)}\n")
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"{'='*70}\n")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=full_env,
    )
    
    for line in iter(proc.stdout.readline, b''):
        line_str = line.decode('utf-8', errors='replace')
        with open(log_file, "a") as f:
            f.write(line_str)
        # Also print to console for monitoring
        print(f"[GPU{gpu_id}] {line_str}", end='')
    
    proc.wait()
    
    with open(log_file, "a") as f:
        f.write(f"\n[Exit code: {proc.returncode}]\n")
    
    return proc.returncode


def run_imagenette_experiments(gpu_id, log_file):
    """Run ImageNette experiments on the given GPU."""
    results = {}
    
    # Experiment 1: ImageNette IPC=10 (AGS-DD full)
    print(f"\n[GPU{gpu_id}] === ImageNette IPC=10 AGS-DD Full ===")
    save_dir = f"{RESULTS_DIR}/imagenette_ipc10_ags"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "nette",
        "--num-samples", "10",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--cags-alpha", "0.5",
        "--cags-beta", "0.5",
        "--iast-lambda", "0.1",
        "--tags-schedule", "cosine",
        "--guidance-scale-min", "0.05",
        "--guidance-scale-max", "0.3",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagenette_ipc10_ags"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 2: ImageNette IPC=10 (MGD3 Baseline)
    print(f"\n[GPU{gpu_id}] === ImageNette IPC=10 MGD3 Baseline ===")
    save_dir = f"{RESULTS_DIR}/imagenette_ipc10_baseline"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "nette",
        "--num-samples", "10",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--baseline",
        "--baseline-stop-t", "0",
        "--baseline-mode-scale", "0.1",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagenette_ipc10_baseline"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 3: ImageNette IPC=50 (AGS-DD full)
    print(f"\n[GPU{gpu_id}] === ImageNette IPC=50 AGS-DD Full ===")
    save_dir = f"{RESULTS_DIR}/imagenette_ipc50_ags"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "nette",
        "--num-samples", "50",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--cags-alpha", "0.5",
        "--cags-beta", "0.5",
        "--iast-lambda", "0.1",
        "--tags-schedule", "cosine",
        "--guidance-scale-min", "0.05",
        "--guidance-scale-max", "0.3",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagenette_ipc50_ags"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 4: ImageNette IPC=50 (MGD3 Baseline)
    print(f"\n[GPU{gpu_id}] === ImageNette IPC=50 MGD3 Baseline ===")
    save_dir = f"{RESULTS_DIR}/imagenette_ipc50_baseline"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "nette",
        "--num-samples", "50",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--baseline",
        "--baseline-stop-t", "0",
        "--baseline-mode-scale", "0.1",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagenette_ipc50_baseline"] = {"exit_code": ret, "save_dir": save_dir}
    
    return results


def run_imagewoof_experiments(gpu_id, log_file):
    """Run ImageWoof experiments on the given GPU."""
    results = {}
    
    # Experiment 1: ImageWoof IPC=10 (AGS-DD full)
    print(f"\n[GPU{gpu_id}] === ImageWoof IPC=10 AGS-DD Full ===")
    save_dir = f"{RESULTS_DIR}/imagewoof_ipc10_ags"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "woof",
        "--num-samples", "10",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagewoof2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--cags-alpha", "0.5",
        "--cags-beta", "0.5",
        "--iast-lambda", "0.1",
        "--tags-schedule", "cosine",
        "--guidance-scale-min", "0.05",
        "--guidance-scale-max", "0.3",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagewoof_ipc10_ags"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 2: ImageWoof IPC=10 (MGD3 Baseline)
    print(f"\n[GPU{gpu_id}] === ImageWoof IPC=10 MGD3 Baseline ===")
    save_dir = f"{RESULTS_DIR}/imagewoof_ipc10_baseline"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "woof",
        "--num-samples", "10",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagewoof2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--baseline",
        "--baseline-stop-t", "0",
        "--baseline-mode-scale", "0.1",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagewoof_ipc10_baseline"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 3: ImageWoof IPC=50 (AGS-DD full)
    print(f"\n[GPU{gpu_id}] === ImageWoof IPC=50 AGS-DD Full ===")
    save_dir = f"{RESULTS_DIR}/imagewoof_ipc50_ags"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "woof",
        "--num-samples", "50",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagewoof2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--cags-alpha", "0.5",
        "--cags-beta", "0.5",
        "--iast-lambda", "0.1",
        "--tags-schedule", "cosine",
        "--guidance-scale-min", "0.05",
        "--guidance-scale-max", "0.3",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagewoof_ipc50_ags"] = {"exit_code": ret, "save_dir": save_dir}
    
    # Experiment 4: ImageWoof IPC=50 (MGD3 Baseline)
    print(f"\n[GPU{gpu_id}] === ImageWoof IPC=50 MGD3 Baseline ===")
    save_dir = f"{RESULTS_DIR}/imagewoof_ipc50_baseline"
    cmd = [
        "python3", "sample_ags.py",
        "--spec", "woof",
        "--num-samples", "50",
        "--nclass", "10",
        "--num-datasets", "3",
        "--imagenet-dir", "/ssd_data/imagenette/imagewoof2/",
        "--save-dir", save_dir,
        "--num-sampling-steps", "50",
        "--closest-point",
        "--baseline",
        "--baseline-stop-t", "0",
        "--baseline-mode-scale", "0.1",
    ]
    ret = run_command(cmd, log_file, gpu_id=gpu_id)
    results["imagewoof_ipc50_baseline"] = {"exit_code": ret, "save_dir": save_dir}
    
    return results


def run_ablation_experiments(gpu_id, log_file):
    """Run ablation studies on the given GPU."""
    results = {}
    
    ablation_configs = [
        ("full", []),
        ("no_cags", ["--no-cags"]),
        ("no_iast", ["--no-iast"]),
        ("no_tags", ["--no-tags"]),
        ("cags_only", ["--no-iast", "--no-tags"]),
        ("iast_only", ["--no-cags", "--no-tags"]),
        ("tags_only", ["--no-cags", "--no-iast"]),
    ]
    
    for name, extra_flags in ablation_configs:
        if name == "full":
            continue  # Skip full since it's already run above
        print(f"\n[GPU{gpu_id}] === Ablation: {name} ===")
        save_dir = f"{RESULTS_DIR}/ablation_nette_ipc10_{name}"
        cmd = [
            "python3", "sample_ags.py",
            "--spec", "nette",
            "--num-samples", "10",
            "--nclass", "10",
            "--num-datasets", "1",
            "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
            "--save-dir", save_dir,
            "--num-sampling-steps", "50",
            "--closest-point",
        ] + extra_flags
        ret = run_command(cmd, log_file, gpu_id=gpu_id)
        results[f"ablation_{name}"] = {"exit_code": ret, "save_dir": save_dir}
    
    return results


def collect_results():
    """Collect all experiment results into a summary."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }
    
    for name in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(path):
            continue
        
        config_file = os.path.join(path, "ags_config.pkl")
        num_images = 0
        num_datasets = 0
        
        for item in os.listdir(path):
            if item.startswith("dataset_"):
                num_datasets += 1
                dataset_path = os.path.join(path, item)
                for cls_dir in os.listdir(dataset_path):
                    cls_path = os.path.join(dataset_path, cls_dir)
                    if os.path.isdir(cls_path):
                        num_images += len([
                            f for f in os.listdir(cls_path) 
                            if f.endswith('.png')
                        ])
        
        summary["experiments"][name] = {
            "num_datasets": num_datasets,
            "num_images": num_images,
            "path": path,
        }
    
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults summary saved to {summary_path}")
    return summary


def upload_to_github(token, log_file):
    """Upload experiment results to GitHub."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"experiment: AGS-DD results {timestamp}"
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"[{timestamp}] Uploading results to GitHub...\n")
    
    cmds = [
        ["git", "add", "results/", "experiment_logs/"],
        ["git", "commit", "-m", msg],
    ]
    
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__)))
        with open(log_file, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{result.stdout}\n{result.stderr}\n")
    
    # Push using token
    push_cmd = [
        "git", "push",
        f"https://{token}@github.com/llmnjust-afk/ICLR2027_KD.git",
        "main"
    ]
    result = subprocess.run(push_cmd, capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
    with open(log_file, "a") as f:
        f.write(f"$ git push\n{result.stdout}\n{result.stderr}\n")
        f.write(f"Push exit code: {result.returncode}\n")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-token", type=str, required=True,
                        help="GitHub PAT for uploading results")
    parser.add_argument("--gpu0-task", type=str, default="imagenette",
                        choices=["imagenette", "imagewoof", "ablation"],
                        help="Task for GPU 0")
    parser.add_argument("--gpu1-task", type=str, default="imagewoof",
                        choices=["imagenette", "imagewoof", "ablation"],
                        help="Task for GPU 1")
    args = parser.parse_args()
    
    log_file = os.path.join(LOG_DIR, "experiment_runner.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a") as f:
        f.write(f"\n{'#'*70}\n")
        f.write(f"# AGS-DD Experiment Run - {timestamp}\n")
        f.write(f"# GPU0: {args.gpu0_task}, GPU1: {args.gpu1_task}\n")
        f.write(f"{'#'*70}\n")
    
    print(f"Starting AGS-DD experiments at {timestamp}")
    print(f"GPU0: {args.gpu0_task}")
    print(f"GPU1: {args.gpu1_task}")
    
    import threading
    
    gpu0_results = {}
    gpu1_results = {}
    
    def gpu0_worker():
        log0 = os.path.join(LOG_DIR, "gpu0.log")
        if args.gpu0_task == "imagenette":
            gpu0_results.update(run_imagenette_experiments(0, log0))
        elif args.gpu0_task == "imagewoof":
            gpu0_results.update(run_imagewoof_experiments(0, log0))
        elif args.gpu0_task == "ablation":
            gpu0_results.update(run_ablation_experiments(0, log0))
    
    def gpu1_worker():
        log1 = os.path.join(LOG_DIR, "gpu1.log")
        if args.gpu1_task == "imagenette":
            gpu1_results.update(run_imagenette_experiments(1, log1))
        elif args.gpu1_task == "imagewoof":
            gpu1_results.update(run_imagewoof_experiments(1, log1))
        elif args.gpu1_task == "ablation":
            gpu1_results.update(run_ablation_experiments(1, log1))
    
    t0 = threading.Thread(target=gpu0_worker)
    t1 = threading.Thread(target=gpu1_worker)
    
    t0.start()
    t1.start()
    
    t0.join()
    t1.join()
    
    # Collect results
    print("\n\nCollecting results...")
    summary = collect_results()
    
    # Upload to GitHub
    print("\nUploading to GitHub...")
    ret = upload_to_github(args.github_token, log_file)
    
    if ret == 0:
        print("Successfully uploaded to GitHub!")
    else:
        print(f"GitHub upload failed (exit code: {ret})")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAll experiments completed at {timestamp}")


if __name__ == "__main__":
    main()
