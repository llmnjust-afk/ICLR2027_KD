"""CLI: prepare -> [tune -> select] -> plan final -> workers -> summarize.

The allocation suite evaluates fixed, prespecified class-allocation controls
without a tuning stage or a claim against the best tuned fixed coefficient.
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

from .common import (ROOT, atomic_json, backup_code, coefficient_vectors, condition_set,
                     digest, file_hash, hardware, now, read_json, read_plan, source_manifest)


def csv_ints(value):
    return [int(x) for x in value.split(",") if x.strip()]


def csv_floats(value):
    return [float(x) for x in value.split(",") if x.strip()]


def mean_matched(values, target, high=.06):
    import numpy as np
    values = np.clip(np.asarray(values) / high, 1e-12, 1 - 1e-12)
    logits = np.log(values / (1 - values))
    lo, hi = -50., 50.
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.mean(high / (1 + np.exp(-(logits + mid)))) < target:
            lo = mid
        else:
            hi = mid
    return (high / (1 + np.exp(-(logits + (lo + hi) / 2)))).tolist()


def validate_prepared(folder):
    manifest = read_json(folder / "manifest.json")
    for rel, checksum in manifest["files"].items():
        if file_hash(folder / rel) != checksum:
            raise ValueError(f"Preparation file changed: {folder / rel}")
    return manifest


def plan_experiment(args):
    workdir = Path(args.workdir).resolve()
    prepared = workdir / "prepared"
    manifest = validate_prepared(prepared)
    if manifest["image_size"] not in {256, 512}:
        raise ValueError("The DiT generation image size must be 256 or 512.")
    if not 0 <= args.guidance_steps <= args.steps:
        raise ValueError("Guidance steps must lie between 0 and denoising steps.")
    if args.precision == "amp" and args.device == "cpu":
        raise ValueError("AMP is an optional CUDA classifier setting.")
    if args.epochs <= 0 or args.generation_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch sizes must be positive.")
    if not args.generation_seeds or len(set(args.generation_seeds)) != len(args.generation_seeds):
        raise ValueError("Generation seeds must be nonempty and unique.")
    if not args.training_seeds or len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("Training seeds must be nonempty and unique.")
    if any(v < 0 for v in args.fixed_grid):
        raise ValueError("Mode-guidance coefficients must be nonnegative.")
    stats = read_json(prepared / "profiles" / f"{args.profile}.json")
    architectures = args.architectures.split(",")
    if not set(architectures) <= {"convnet6", "resnet18", "resnetap10"}:
        raise ValueError("Architectures: convnet6,resnet18,resnetap10")
    selection, best = None, None
    if args.suite in {"core", "full"}:
        if not args.selection:
            raise ValueError("Use --selection from a completed held-out tuning run.")
        selection = read_json(args.selection)
        if selection["prepared_manifest_hash"] != file_hash(prepared / "manifest.json"):
            raise ValueError("Selection used another preparation/data split.")
        if selection["profile"] != args.profile:
            raise ValueError("Selection and final experiments must use the same profile.")
        if set(args.generation_seeds) & set(selection["generation_seeds"]):
            raise ValueError("Final generation seeds must be disjoint from tuning seeds.")
        for arch in architectures:
            if arch not in selection["best_fixed_by_arch"]:
                raise ValueError(f"Tune {arch} first, or run final evaluation only for tuned architectures.")
        best = selection["best_fixed_by_arch"][architectures[0]]["coefficient"]
    conditions = condition_set(stats, args.suite, args.fixed_grid, args.permutation_seeds, best)
    selected_references = {}
    if args.suite in {"core", "full"}:
        conditions.pop("fixed_best")
        for arch in architectures:
            value = selection["best_fixed_by_arch"][arch]["coefficient"]
            conditions[f"fixed_best_{arch}"] = {"coefficients": [value] * len(stats["classes"]),
                                                "kind": "fixed", "value": value, "only_arch": arch}
            params = selection["best_cags_by_arch"][arch]["hyperparameters"]
            values = coefficient_vectors(stats, high=params["maximum"], slope=params["slope"],
                                         center=params["center"], candidate_weights=params["weights"])["candidate"]
            if values == conditions["cags"]["coefficients"]:
                selected_references[arch] = "cags"
                continue
            reference = f"cags_selected_{arch}"
            selected_references[arch] = reference
            conditions[reference] = {"coefficients": values, "kind": "heldout_selected_cags",
                                     "only_arch": arch, "hyperparameters": params}
            conditions[f"selected_fixed_mean_{arch}"] = {"coefficients": [sum(values) / len(values)] * len(values),
                                                         "kind": "selected_fixed_mean", "only_arch": arch}
            import numpy as np
            for seed in args.permutation_seeds:
                order = np.random.default_rng(seed).permutation(len(values)).tolist()
                conditions[f"selected_permuted_{arch}_{seed}"] = {
                    "coefficients": [values[i] for i in order], "kind": "permuted", "only_arch": arch,
                    "permutation_seed": seed, "source_class_indices": order, "permutation_of": reference}
    target_mean = sum(conditions["cags"]["coefficients"]) / len(stats["classes"])
    if args.suite in {"factors", "full"}:
        for name in ["entropy", "dispersion", "four_factor"]:
            conditions[name + "_mean_matched"] = {"coefficients": mean_matched(conditions[name]["coefficients"], target_mean),
                "kind": "mean_matched_factor", "factor": name, "matching": "logit intercept shifted to CAGS mean"}
    if args.suite == "sensitivity":
        for name in manifest["profiles"]:
            if name == args.profile:
                continue
            profile = read_json(prepared / "profiles" / f"{name}.json")
            values = coefficient_vectors(profile)["cags"]
            conditions[name] = {"coefficients": values, "kind": "estimator_sensitivity", "profile": name}
            conditions[name + "_mean_matched"] = {"coefficients": mean_matched(values, target_mean),
                                                    "kind": "estimator_sensitivity", "profile": name,
                                                    "matching": "logit intercept shifted to CAGS mean"}
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Set --checkpoint to the downloaded DiT-XL/2 checkpoint: {checkpoint}")
    from .evaluate import evaluation_paths, image_manifest
    eval_split = "tune" if args.suite == "tune" else "report"
    if eval_split == "report" and not args.report_dir:
        raise ValueError("Final experiments require --report-dir /path/to/imagenet/val.")
    eval_root, eval_paths = evaluation_paths(prepared, args.report_dir, eval_split)
    protocol = {"prepared": str(prepared), "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
                "vae": args.vae or manifest["vae"], "local_files_only": args.local_files_only,
                "device": args.device, "image_size": manifest["image_size"], "ipc": manifest["ipc"],
                "steps": args.steps, "guidance_steps": args.guidance_steps, "cfg_scale": args.cfg_scale,
                "generation_batch_size": args.generation_batch_size, "deterministic": args.deterministic,
                "epochs": args.epochs, "batch_size": args.batch_size, "classifier_size": args.classifier_size,
                "precision": args.precision, "lr": args.lr, "weight_decay": args.weight_decay,
                "threads": args.threads, "num_workers": args.num_workers, "save_checkpoints": args.save_checkpoints,
                "checkpoint_rule": "final" if eval_split == "tune" else args.checkpoint_rule,
                "eval_interval": args.eval_interval, "evaluation_split": eval_split,
                "evaluation_root": eval_root, "evaluation_image_manifest": image_manifest(eval_root, eval_paths)}
    protocol["guidance_definition"] = "Adaptive mode-attraction coefficient, separate from fixed classifier-free guidance"
    protocol["coefficient_formula"] = "maximum * sigmoid(slope * (weighted_normalized_factors - center)); per-condition parameters/vectors are authoritative"
    protocol["latent_scaling"] = .18215
    if selection:
        for key in ["steps", "guidance_steps", "cfg_scale", "generation_batch_size", "epochs", "batch_size",
                    "classifier_size", "precision", "lr", "weight_decay", "checkpoint_sha256", "deterministic"]:
            if protocol[key] != selection["tuning_protocol"][key]:
                raise ValueError(f"Final/tuning protocol differs in {key}; create a matching tuning experiment.")
    plan = {"schema": 1, "suite": args.suite, "classes": manifest["classes"], "profile": args.profile,
            "prepared_manifest": manifest, "prepared_manifest_hash": file_hash(prepared / "manifest.json"),
            "generation_seeds": args.generation_seeds, "training_seeds": args.training_seeds,
            "architectures": architectures, "conditions": conditions, "protocol": protocol,
            "source": source_manifest(), "selection": selection, "environment": hardware()["versions"],
            "selected_cags_references": selected_references}
    plan["plan_hash"] = digest(plan)
    folder = workdir / "runs" / args.name
    if (folder / "plan.json").exists():
        if read_plan(folder / "plan.json")["plan_hash"] != plan["plan_hash"]:
            raise ValueError(f"Run name already has a different immutable plan: {folder}; choose another --name.")
        if not (folder / "READY.json").exists():
            raise ValueError(f"Incomplete plan creation at {folder}; archive it and use another --name.")
        print(f"Reusing identical plan: {folder}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "plan.json", plan)
    atomic_json(folder / "evaluation_images.json", {"root": eval_root, "paths": eval_paths})
    queue = folder / "queue"
    queue.mkdir(exist_ok=True)
    for seed in args.generation_seeds:
        for name in conditions:
            atomic_json(queue / f"gen_g{seed}_{name}.job.json",
                        {"stage": "generate", "generation_seed": seed, "condition": name})
        for training in args.training_seeds:
            for arch in architectures:
                atomic_json(queue / f"eval_g{seed}_t{training}_{arch}.job.json",
                            {"stage": "evaluate", "generation_seed": seed, "training_seed": training, "architecture": arch})
    archive = backup_code(folder / "code_backup")
    atomic_json(folder / "READY.json", {"plan_hash": plan["plan_hash"], "created": now()})
    counts = {arch: sum(c.get("only_arch", arch) == arch for c in conditions.values()) for arch in architectures}
    print(json.dumps({"plan": str(folder), "source_backup": str(archive), "conditions": list(conditions),
                      "generated_datasets": len(conditions) * len(args.generation_seeds),
                      "classifier_fits": sum(counts.values()) * len(args.generation_seeds) * len(args.training_seeds)}, indent=2))


def job_state(job):
    base = str(job)[:-len(".job.json")]
    for suffix, state in [(".done.json", "done"), (".failed.json", "failed"), (".lock", "running")]:
        if Path(base + suffix).exists():
            return state
    return "pending"


def claim_job(job):
    base = str(job)[:-len(".job.json")]
    if job_state(job) != "pending":
        return None
    lock = Path(base + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "created": now()}, stream)
    # Recheck terminal states in case another worker completed between the first check and lock acquisition.
    if Path(base + ".done.json").exists() or Path(base + ".failed.json").exists():
        lock.unlink()
        return None
    return lock


def execute_evaluation(plan, plan_dir, job):
    from .evaluate import evaluation_paths, image_manifest, loader, train_one
    from .generate import dataset_dir, verify_dataset
    cfg = plan["protocol"]
    data = read_json(plan_dir / "evaluation_images.json")
    if data["root"] != cfg["evaluation_root"] or image_manifest(data["root"], data["paths"]) != cfg["evaluation_image_manifest"]:
        raise ValueError("Evaluation image list or source files changed after planning.")
    batches = loader(data["root"], data["paths"], cfg["classifier_size"], cfg["batch_size"], cfg["num_workers"])
    tuning = None
    if cfg["checkpoint_rule"] == "tune_best":
        root, paths = evaluation_paths(Path(cfg["prepared"]), None, "tune")
        tuning = loader(root, paths, cfg["classifier_size"], cfg["batch_size"], cfg["num_workers"])
    output_dir = plan_dir / "evaluation" / f"g{job['generation_seed']}" / f"t{job['training_seed']}" / job["architecture"]
    for name, condition in plan["conditions"].items():
        if condition.get("only_arch", job["architecture"]) != job["architecture"]:
            continue
        output = output_dir / f"{name}.json"
        if output.exists():
            existing = read_json(output)
            if existing["plan_hash"] != plan["plan_hash"]:
                raise ValueError(f"Stale evaluation at {output}")
            if existing["hardware"]["gpu"] != hardware()["gpu"]:
                raise ValueError("Resume this paired training block on the same GPU model, or use a new training-seed block.")
            continue
        folder = dataset_dir(plan_dir, job["generation_seed"], name)
        expected = digest({"plan_hash": plan["plan_hash"], "generation_seed": job["generation_seed"],
                           "condition": name, "specification": condition})
        verify_dataset(folder, expected)
        provenance = {"plan_hash": plan["plan_hash"], "condition": name, "generation_seed": job["generation_seed"],
                      "generation_manifest_sha256": file_hash(folder / "manifest.json"),
                      "prepared_manifest_sha256": plan["prepared_manifest_hash"]}
        train_one(folder, plan["classes"], cfg, job["training_seed"], job["architecture"],
                  batches, tuning, output, provenance)


def bind_generation_gpu(plan_dir, generation_seed, current_hardware):
    """Atomically bind a paired generation block to one GPU model across hosts."""
    block = plan_dir / "generation_hardware" / f"g{generation_seed}.json"
    block.parent.mkdir(exist_ok=True)
    if not block.exists():
        temp = block.with_name(block.name + f".{socket.gethostname()}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(current_hardware))
        try:
            os.link(temp, block)
        except FileExistsError:
            pass
        finally:
            temp.unlink(missing_ok=True)
    return read_json(block)["gpu"] == current_hardware["gpu"]


def generation_complete(plan_dir, plan, seed):
    return all((plan_dir / "queue" / f"gen_g{seed}_{name}.done.json").exists()
               for name in plan["conditions"])


def worker(args):
    import signal
    def stop_worker(signum, frame):
        raise KeyboardInterrupt("Worker terminated; partial job will be marked failed and unlocked.")
    signal.signal(signal.SIGTERM, stop_worker)
    from .generate import GenerationEngine
    plan_dir = Path(args.plan).resolve()
    plan = read_plan(plan_dir / "plan.json")
    if not (plan_dir / "READY.json").exists():
        raise ValueError("Plan creation has not completed; wait for the planning command to finish.")
    if source_manifest() != plan["source"]:
        raise ValueError("Source code changed after planning. Finish this plan with its archived code, or make a new run.")
    if hardware()["versions"] != plan["environment"]:
        raise ValueError("Worker package versions differ from the frozen plan environment.")
    if file_hash(Path(plan["protocol"]["prepared"]) / "manifest.json") != plan["prepared_manifest_hash"]:
        raise ValueError("Preparation manifest changed after planning.")
    prepared = Path(plan["protocol"]["prepared"])
    for rel in ["mode_targets.npz", "split.json"]:
        if file_hash(prepared / rel) != plan["prepared_manifest"]["files"][rel]:
            raise ValueError(f"Prepared generation/evaluation input changed: {rel}")
    if file_hash(plan["protocol"]["checkpoint"]) != plan["protocol"]["checkpoint_sha256"]:
        raise ValueError("DiT checkpoint changed after planning.")
    jobs = sorted(p for p in (plan_dir / "queue").glob("*.job.json") if read_json(p)["stage"] == args.stage)
    engine, failed = None, False
    current_hardware = hardware()
    try:
        while True:
            pending = [p for p in jobs if job_state(p) == "pending"]
            if not pending:
                if any(job_state(p) == "running" for p in jobs):
                    time.sleep(2)
                    continue
                failed = failed or any(job_state(p) == "failed" for p in jobs)
                break
            made_progress = False
            for path in pending:
                job = read_json(path)
                if args.stage == "evaluate":
                    if not generation_complete(plan_dir, plan, job["generation_seed"]):
                        continue
                elif not bind_generation_gpu(plan_dir, job["generation_seed"], current_hardware):
                    continue
                lock = claim_job(path)
                if lock is None:
                    continue
                made_progress = True
                base = str(path)[:-len(".job.json")]
                print(f"START {path.name} on {hardware()['gpu']}", flush=True)
                try:
                    if source_manifest() != plan["source"]:
                        raise ValueError("Source changed while workers were running; use a new frozen experiment plan.")
                    if args.stage == "generate":
                        if engine is None:
                            engine = GenerationEngine(plan)
                        name = job["condition"]
                        engine.generate(plan, plan_dir, job["generation_seed"], name, plan["conditions"][name])
                    else:
                        execute_evaluation(plan, plan_dir, job)
                    atomic_json(base + ".done.json", {"job": job, "finished": now(), "hardware": hardware(), "plan_hash": plan["plan_hash"]})
                except BaseException:
                    failed = True
                    atomic_json(base + ".failed.json", {"job": job, "failed": now(), "traceback": traceback.format_exc(), "hardware": hardware()})
                    traceback.print_exc()
                    if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):
                        raise
                finally:
                    lock.unlink(missing_ok=True)
                break
            if failed:
                break
            if not made_progress:
                if args.stage == "generate":
                    # Other GPU models may own the remaining paired blocks. This
                    # worker must not change their hardware halfway through.
                    break
                # Never silently run a classifier on a partial or failed generation.
                if args.stage == "evaluate" and any(job_state(p) == "pending" for p in jobs):
                    generation = list((plan_dir / "queue").glob("gen_*.job.json"))
                    if not any(job_state(p) in {"pending", "running"} for p in generation):
                        failed = True
                        break
                time.sleep(2)
    finally:
        if engine is not None:
            engine.close()
    if failed:
        raise SystemExit("Some jobs failed or have failed dependencies. See queue/*.failed.json; then retry-failed.")


def select_fixed(args):
    from .statistics import summarize
    folder = Path(args.plan).resolve()
    plan = read_plan(folder / "plan.json")
    if plan["suite"] != "tune" or plan["protocol"]["evaluation_split"] != "tune":
        raise ValueError("Selection is allowed only from a completed tuning-split experiment.")
    summary = summarize(folder)
    best, best_cags = {}, {}
    for arch, data in summary["by_architecture"].items():
        candidates = []
        for name, result in data["conditions"].items():
            condition = plan["conditions"][name]
            value = 0.0 if condition["kind"] == "no_mode" else condition.get("value")
            if value is not None and condition["kind"] in {"no_mode", "fixed"}:
                candidates.append((result["mean_top1"], -value, name, value))
        if not candidates:
            raise ValueError("No prespecified fixed coefficient candidates.")
        score, _, name, value = max(candidates)
        best[arch] = {"condition": name, "coefficient": value, "tuning_mean_top1": score}
        adaptive = []
        for candidate, result in data["conditions"].items():
            specification = plan["conditions"][candidate]
            if specification["kind"] in {"cags", "cags_candidate"}:
                params = specification["hyperparameters"]
                adaptive.append((result["mean_top1"], sum(x != 0 for x in params["weights"]), candidate, params))
        winner = min(adaptive, key=lambda value: (-value[0], value[1], value[2]))
        best_cags[arch] = {"condition": winner[2], "hyperparameters": winner[3], "tuning_mean_top1": winner[0]}
    selection = {"source_plan_hash": plan["plan_hash"], "prepared_manifest_hash": plan["prepared_manifest_hash"],
                 "profile": plan["profile"], "generation_seeds": plan["generation_seeds"],
                 "best_fixed_by_arch": best, "best_cags_by_arch": best_cags, "tuning_protocol": plan["protocol"],
                 "selection_rule": "All prespecified fixed coefficients including zero and 0.10 are eligible; maximize held-out mean, fixed ties use smaller coefficient; CAGS ties use fewer nonzero weights then lexical name",
                 "all_tuning_results": summary["by_architecture"]}
    output = Path(args.output or folder / "selection.json")
    if output.exists() and read_json(output) != selection:
        raise ValueError(f"Refusing to replace a different frozen selection: {output}")
    atomic_json(output, selection)
    print(json.dumps({"best_fixed": best, "best_cags": best_cags}, indent=2))
    print(f"Frozen selection: {output}")


def status(args):
    from collections import Counter
    queue = Path(args.plan) / "queue"
    for stage in ["generate", "evaluate"]:
        jobs = [p for p in queue.glob("*.job.json") if read_json(p)["stage"] == stage]
        print(stage, dict(Counter(job_state(p) for p in jobs)))
    for path in sorted(queue.glob("*.failed.json")):
        print("FAILED", path.name, read_json(path)["traceback"].splitlines()[-1])
    for path in sorted(queue.glob("*.lock")):
        print("LOCK", path.name, path.read_text())


def retry_failed(args):
    for path in (Path(args.plan) / "queue").glob("*.failed.json"):
        base = str(path)[:-len(".failed.json")]
        if Path(base + ".lock").exists():
            raise ValueError(f"Job still locked: {base}; verify its worker has exited.")
        archive = path.with_name(path.name + "." + str(time.time_ns()) + ".history")
        path.rename(archive)
        print("Requeued", base)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create an independent tuning split and immutable caches")
    prep.add_argument("--workdir", required=True)
    prep.add_argument("--data-root", required=True, help="Contains train/WNID and val/WNID")
    prep.add_argument("--class-file", default=str(ROOT / "misc/class100.txt"))
    prep.add_argument("--nclass", type=int, default=100)
    prep.add_argument("--ipc", type=int, default=10)
    prep.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    prep.add_argument("--image-size", type=int, default=256)
    prep.add_argument("--holdout-fraction", type=float, default=.1)
    prep.add_argument("--seed", type=int, default=2027)
    prep.add_argument("--feature-batch-size", type=int, default=16)
    prep.add_argument("--num-workers", type=int, default=4)
    prep.add_argument("--pca-components", type=int, default=4)
    prep.add_argument("--k-max", type=int, default=20)
    prep.add_argument("--k-values", type=csv_ints, default=[2, 5, 10])
    prep.add_argument("--pca-variants", type=csv_ints, default=[])
    prep.add_argument("--resnet-checkpoint", help="Optional torchvision ResNet-18 ImageNet state_dict for feature sensitivity")
    prep.add_argument("--feature-flip", action="store_true", help="Deterministic per-image flips; default is center-crop only")
    prep.add_argument("--local-files-only", action="store_true")
    prep.add_argument("--deterministic", action="store_true")
    prep.add_argument("--device", default="cuda")
    p = sub.add_parser("plan", help="Freeze an experiment and create a shared-filesystem work queue")
    p.add_argument("--workdir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vae")
    p.add_argument("--report-dir")
    p.add_argument("--suite", choices=["allocation", "tune", "core", "full", "factors", "sensitivity", "cfg_controls"], required=True)
    p.add_argument("--selection")
    p.add_argument("--profile", default="vae_auto")
    p.add_argument("--generation-seeds", type=csv_ints, default=[100, 101, 102, 103, 104])
    p.add_argument("--training-seeds", type=csv_ints, default=[0, 1, 2])
    p.add_argument("--permutation-seeds", type=csv_ints, default=[17, 29, 43])
    p.add_argument("--fixed-grid", type=csv_floats,
                   default=[.005, .01, .015, .02, .025, .03, .035, .04, .045, .05, .06, .08, .10])
    p.add_argument("--architectures", default="convnet6")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-steps", type=int, default=25)
    p.add_argument("--cfg-scale", type=float, default=4)
    p.add_argument("--generation-batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--classifier-size", type=int, default=224)
    p.add_argument("--checkpoint-rule", choices=["final", "tune_best"], default="final")
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    p.add_argument("--lr", type=float, default=.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", default="cuda")
    w = sub.add_parser("worker", help="One worker per visible GPU, shared queue across machines")
    w.add_argument("--plan", required=True)
    w.add_argument("--stage", choices=["generate", "evaluate"], required=True)
    for name in ["status", "retry-failed"]:
        q = sub.add_parser(name)
        q.add_argument("--plan", required=True)
    s = sub.add_parser("select", help="Freeze the best fixed coefficient using held-out tuning only")
    s.add_argument("--plan", required=True)
    s.add_argument("--output")
    s = sub.add_parser("summarize")
    s.add_argument("--plan", required=True)
    s.add_argument("--allow-partial", action="store_true")
    sub.add_parser("backup")
    smoke = sub.add_parser("smoke", help="GPU/model plumbing check only; excluded from scientific summaries")
    smoke.add_argument("--workdir", required=True)
    smoke.add_argument("--checkpoint", required=True)
    smoke.add_argument("--vae", required=True)
    smoke.add_argument("--generation-batch-size", type=int, default=4)
    smoke.add_argument("--classifier-batch-size", type=int, default=128)
    smoke.add_argument("--local-files-only", action="store_true")
    smoke.add_argument("--deterministic", action="store_true")
    return root


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from .prepare import prepare
        prepare(args)
    elif args.command == "plan":
        if not args.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Run name may contain letters, digits, underscores and hyphens only.")
        plan_experiment(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "select":
        select_fixed(args)
    elif args.command == "status":
        status(args)
    elif args.command == "retry-failed":
        retry_failed(args)
    elif args.command == "backup":
        print(backup_code())
    elif args.command == "smoke":
        from .smoke import smoke
        smoke(args)
    elif args.command == "summarize":
        from .statistics import summarize
        summary = summarize(args.plan, args.allow_partial)
        print(json.dumps(summary["by_architecture"], indent=2))


if __name__ == "__main__":
    main()
