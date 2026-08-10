"""
AGS-DD Ablation Study Runner

Runs all ablation configurations for the paper:
  1. Baseline (MGD3 equivalent: no CAGS, no IAST, no TAGS)
  2. + CAGS only
  3. + IAST only
  4. + TAGS only
  5. + CAGS + IAST
  6. + CAGS + TAGS
  7. + IAST + TAGS
  8. Full (CAGS + IAST + TAGS)

Usage:
    python run_ablation.py --spec nette --num-samples 10 --test-dir /path/to/test
"""

import os
import sys
import argparse
import json
import time
import subprocess
import numpy as np

from evaluation import AblationResults, DistillationMetrics


ABLATION_CONFIGS = [
    {"name": "baseline",      "cags": False, "iast": False, "tags": False},
    {"name": "cags_only",     "cags": True,  "iast": False, "tags": False},
    {"name": "iast_only",     "cags": False, "iast": True,  "tags": False},
    {"name": "tags_only",      "cags": False, "iast": False, "tags": True},
    {"name": "cags_iast",      "cags": True,  "iast": True,  "tags": False},
    {"name": "cags_tags",      "cags": True,  "iast": False, "tags": True},
    {"name": "iast_tags",      "cags": False, "iast": True,  "tags": True},
    {"name": "full_ags_dd",    "cags": True,  "iast": True,  "tags": True},
]


def run_ablation_config(config, base_args, output_base):
    """Run a single ablation configuration."""
    name = config["name"]
    save_dir = os.path.join(output_base, name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python", "sample_ags.py",
        "--spec", base_args["spec"],
        "--num-samples", str(base_args["num_samples"]),
        "--save-dir", save_dir,
        "--num-datasets", str(base_args.get("num_datasets", 5)),
        "--imagenet-dir", base_args.get("imagenet_dir", "/ssd_data/imagenet/"),
    ]

    if not config["cags"]:
        cmd.append("--no-cags")
    if not config["iast"]:
        cmd.append("--no-iast")
    if not config["tags"]:
        cmd.append("--no-tags")

    if base_args.get("ckpt"):
        cmd.extend(["--ckpt", base_args["ckpt"]])

    print(f"\n{'='*70}")
    print(f"Running ablation: {name}")
    print(f"  CAGS: {config['cags']}, IAST: {config['iast']}, TAGS: {config['tags']}")
    print(f"  Output: {save_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    return {
        "name": name,
        "config": config,
        "time_s": elapsed,
        "stdout": result.stdout[-500:] if result.stdout else "",
        "stderr": result.stderr[-500:] if result.stderr else "",
        "returncode": result.returncode,
    }


def evaluate_ablation_config(config, base_args, output_base, test_dir, arch="convnet7"):
    """Evaluate a single ablation configuration."""
    name = config["name"]
    train_dir = os.path.join(output_base, name, "dataset_0")

    if not os.path.isdir(train_dir):
        print(f"  Warning: {train_dir} does not exist, skipping evaluation.")
        return None

    cmd = [
        "python", "-m", "evaluation.train_eval",
        "--train-dir", train_dir,
        "--test-dir", test_dir,
        "--arch", arch,
        "--num-classes", str(base_args.get("num_classes", 10)),
        "--epochs", str(base_args.get("epochs", 2000)),
        "--num-seeds", str(base_args.get("num_seeds", 5)),
        "--save-dir", os.path.join(output_base, "eval_results"),
        "--dataset-name", f"ablation_{name}",
        "--ipc", str(base_args["num_samples"]),
    ]

    print(f"\n  Evaluating {name} with {arch}...")
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Load result
    result_path = os.path.join(
        output_base, "eval_results",
        f"ablation_{name}_ipc{base_args['num_samples']}_{arch}.json"
    )

    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            eval_result = json.load(f)
        return {
            "name": name,
            "eval_result": eval_result,
            "eval_time_s": elapsed,
        }
    else:
        print(f"  Warning: Result file not found at {result_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="AGS-DD Ablation Study")
    parser.add_argument("--spec", type=str, default="nette",
                        choices=["nette", "woof", "imagenet100", "imagenet1k"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output-base", type=str, default="./ablation")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to real test dataset")
    parser.add_argument("--arch", type=str, default="convnet7")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-datasets", type=int, default=5)
    parser.add_argument("--imagenet-dir", type=str, default="/ssd_data/imagenet/")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--configs", type=str, nargs="*",
                        default=None,
                        help="Specific configs to run (default: all)")

    args = parser.parse_args()
    os.makedirs(args.output_base, exist_ok=True)

    base_args = {
        "spec": args.spec,
        "num_samples": args.num_samples,
        "num_classes": args.num_classes,
        "num_datasets": args.num_datasets,
        "imagenet_dir": args.imagenet_dir,
        "ckpt": args.ckpt,
        "epochs": args.epochs,
        "num_seeds": args.num_seeds,
    }

    # Filter configs if specified
    configs = ABLATION_CONFIGS
    if args.configs:
        configs = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]

    # Run generation
    gen_results = []
    if not args.skip_generation:
        for config in configs:
            result = run_ablation_config(config, base_args, args.output_base)
            gen_results.append(result)

    # Run evaluation
    eval_results = []
    for config in configs:
        result = evaluate_ablation_config(
            config, base_args, args.output_base, args.test_dir, args.arch
        )
        if result:
            eval_results.append(result)

    # Build ablation summary
    ablation = AblationResults()
    for config in configs:
        ablation.add_config(config["name"], config["cags"], config["iast"], config["tags"])

    for result in eval_results:
        name = result["name"]
        eval_data = result["eval_result"]
        if "top1_stats" in eval_data:
            stats = eval_data["top1_stats"]
            ablation.add_result(
                name, args.spec, args.num_samples, args.arch,
                stats["mean"], stats["std"]
            )

    # Print and save ablation table
    table = ablation.format_table(args.spec, args.num_samples, args.arch)
    print(f"\n{table}")

    summary_path = os.path.join(args.output_base, "ablation_summary.json")
    summary = {
        "dataset": args.spec,
        "ipc": args.num_samples,
        "arch": args.arch,
        "gen_results": gen_results,
        "eval_results": eval_results,
        "table": table,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAblation study complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
