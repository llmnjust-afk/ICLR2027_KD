"""GPU-only plumbing check. Its toy targets/results never enter scientific summaries."""
from datetime import datetime, timezone
from pathlib import Path
import math
import os

import numpy as np
import torch

from .common import ROOT, atomic_json, digest, file_hash, hardware
from .prepare import parameter_hash


def smoke(args):
    from diffusers import AutoencoderKL
    from .generate import GenerationEngine, dataset_dir
    from .evaluate import loader, train_one
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires a visible NVIDIA GPU. Run this command on a GPU node.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path(args.workdir).resolve() / "smoke_tests" / stamp
    prepared = folder / "toy_prepared"
    prepared.mkdir(parents=True)
    classes = (ROOT / "misc/class100.txt").read_text().splitlines()[:2]
    all_classes = (ROOT / "misc/class_indices.txt").read_text().splitlines()
    vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=args.local_files_only)
    vae_hash = parameter_hash(vae)
    del vae
    ipc = args.generation_batch_size
    np.savez(prepared / "mode_targets.npz", **{cls: np.zeros((ipc, 4096), dtype=np.float32) for cls in classes})
    cfg = {"prepared": str(prepared), "checkpoint": str(Path(args.checkpoint).resolve()),
           "vae": args.vae, "device": "cuda", "image_size": 256, "ipc": ipc,
           "steps": 50, "guidance_steps": 25, "cfg_scale": 4., "generation_batch_size": args.generation_batch_size,
           "deterministic": args.deterministic, "local_files_only": args.local_files_only,
           "threads": 2, "classifier_size": 224, "epochs": 1, "batch_size": 8,
           "precision": "fp32", "lr": .1, "weight_decay": 1e-4, "checkpoint_rule": "final",
           "eval_interval": 1, "save_checkpoints": True}
    plan = {"protocol": cfg, "classes": classes,
            "prepared_manifest": {"imagenet_labels": [all_classes.index(c) for c in classes],
                                  "vae_parameter_sha256": vae_hash},
            "plan_hash": digest({"smoke_only": stamp, "checkpoint": file_hash(args.checkpoint)})}
    torch.cuda.reset_peak_memory_stats()
    engine = GenerationEngine(plan)
    generated = {}
    try:
        for name, value in [("no_mode", 0.), ("mode_003", .03)]:
            generated[name] = engine.generate(plan, folder, 0, name, {"coefficients": [value, value]})
    finally:
        engine.close()
    generation_peak = torch.cuda.max_memory_allocated() / 2**30
    if generated["no_mode"]["noise_seeds"] != generated["mode_003"]["noise_seeds"]:
        raise AssertionError("Paired generation stream IDs differ")
    synthetic = dataset_dir(folder, 0, "mode_003")
    paths = [(f"{cls}/{i:04d}.png", label) for label, cls in enumerate(classes) for i in range(ipc)]
    # Repeat diagnostic images via hard links to check the real requested batch's
    # activation footprint. This remains a plumbing test, never an accuracy experiment.
    repeated = folder / "diagnostic_repeated_images"
    training_ipc = max(ipc, math.ceil(args.classifier_batch_size / 2))
    for cls in classes:
        (repeated / cls).mkdir(parents=True)
        for i in range(training_ipc):
            os.link(synthetic / cls / f"{i % ipc:04d}.png", repeated / cls / f"{i:04d}.png")
    train_cfg = dict(cfg, ipc=training_ipc, batch_size=args.classifier_batch_size)
    torch.cuda.reset_peak_memory_stats()
    result = train_one(repeated, classes, train_cfg, 0, "convnet6", loader(synthetic, paths, 224, 8, 0),
                       None, folder / "diagnostic_training.json",
                       {"smoke_test_only": True, "plan_hash": plan["plan_hash"],
                        "note": "Toy zero mode targets; self-evaluation only. Not a dataset-distillation accuracy result."})
    atomic_json(folder / "SMOKE_PASSED.json", {"smoke_test_only": True, "hardware": hardware(),
                                               "diagnostic_result": result,
                                               "generation_batch_size": args.generation_batch_size,
                                               "classifier_batch_size": args.classifier_batch_size,
                                               "peak_generation_allocated_gib": generation_peak,
                                               "peak_training_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                               "note": "Do not include smoke images or accuracies in rebuttal tables."})
    print(f"GPU generation, paired stream IDs, classifier batch={args.classifier_batch_size} passed: {folder}")
