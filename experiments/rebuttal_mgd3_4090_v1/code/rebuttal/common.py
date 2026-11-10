"""Small provenance and configuration helpers; no GPU imports."""
import hashlib
import json
import os
import platform
import random
import socket
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_plan(path):
    plan = read_json(path)
    if digest({k: v for k, v in plan.items() if k != "plan_hash"}) != plan.get("plan_hash"):
        raise ValueError(f"Frozen plan content/hash mismatch: {path}")
    return plan


def seed_for(*parts):
    return int(digest(list(parts))[:15], 16) % (2**31 - 1)


def source_files():
    paths = [ROOT / "models.py", ROOT / "quick_eval_v3.py", ROOT / "download.py"]
    for folder in ["rebuttal", "ags", "diffusion", "train_models", "tests"]:
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    return sorted(set(paths))


def source_manifest():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}


def backup_code(destination=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path(destination or ROOT / "backups") / f"rebuttal_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for rel in manifest:
            archive.add(ROOT / rel, arcname=rel, recursive=False)
        for name in [".gitignore", "scripts/rebuttal.sh", "scripts/run_rebuttal_all.sh",
                     "scripts/submit_rebuttal.sh", "scripts/rebuttal_slurm_job.sh",
                     "scripts/submit_rebuttal_core.sh", "scripts/prepare_rebuttal_core.sh", "rebuttal.core.env",
                     "scripts/submit_rebuttal_tuned.sh", "scripts/submit_rebuttal_campaign.sh", "rebuttal.tuned.env",
                     "rebuttal.example.env", "REBUTTAL_RUN_GUIDE.txt", "REBUTTAL_CORE_NOTES.txt"]:
            if (ROOT / name).exists():
                archive.add(ROOT / name, arcname=name, recursive=False)
                manifest[name] = file_hash(ROOT / name)
    atomic_json(folder / "manifest.json", {"created": now(), "files": manifest})
    return folder


def seed_everything(seed, deterministic=False):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def hardware():
    import torch
    from importlib.metadata import version
    return {"host": socket.gethostname(), "python": platform.python_version(),
            "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "versions": {k: version(k) for k in ["torch", "torchvision", "numpy", "scikit-learn", "diffusers"]}}


def coefficient_vectors(stats, low=0.0, high=0.06, slope=3.0, center=0.6, candidate_weights=None):
    import numpy as np
    names = stats["classes"]
    records = [stats["per_class"][c] for c in names]
    variance = np.array([r["dispersion"] for r in records])
    normalized = (variance - variance.min()) / max(float(np.ptp(variance)), 1e-8)
    entropy = np.array([r["entropy_normalized"] for r in records])
    modes = np.array([r["k"] / stats["k_max"] for r in records])
    sep = np.array([r["separability"] for r in records])
    weights = {"cags": (0, .5, .5, 0), "entropy": (0, 1, 0, 0),
               "dispersion": (0, 0, 1, 0), "four_factor": (.3, .3, .2, .2)}
    if candidate_weights is not None:
        weights["candidate"] = candidate_weights
    out = {}
    for key, (a, b, g, d) in weights.items():
        raw = a * modes + b * entropy + g * normalized + d * (1 - sep)
        out[key] = (low + (high - low) / (1 + np.exp(-slope * (raw - center)))).tolist()
    return out


def tuning_candidates(stats):
    base = {"weights": [0, .5, .5, 0], "slope": 3., "center": .6, "maximum": .06}
    variants = {}
    for name, weights in [("entropy", [0, 1, 0, 0]), ("dispersion", [0, 0, 1, 0]),
                          ("four_factor", [.3, .3, .2, .2]), ("h25_v75", [0, .25, .75, 0]),
                          ("h75_v25", [0, .75, .25, 0])]:
        variants[name] = dict(base, weights=weights)
    for value in [2., 5.]:
        variants[f"slope{int(value)}"] = dict(base, slope=value)
    for value in [.4, .5, .7]:
        variants[f"center{int(value * 10)}"] = dict(base, center=value)
    for value in [.04, .08]:
        variants[f"max{int(value * 100)}"] = dict(base, maximum=value)
    return {f"cags_tune_{name}": {"kind": "cags_candidate", "hyperparameters": params,
            "coefficients": coefficient_vectors(stats, high=params["maximum"], slope=params["slope"],
                                                center=params["center"], candidate_weights=params["weights"])["candidate"]}
            for name, params in variants.items()}


def condition_set(stats, suite, fixed_grid, permutations, best_fixed=None):
    import numpy as np
    vectors = coefficient_vectors(stats)
    count = len(stats["classes"])
    result = {"no_mode": {"coefficients": [0.0] * count, "kind": "no_mode"},
              "cags": {"coefficients": vectors["cags"], "kind": "cags",
                       "hyperparameters": {"weights": [0, .5, .5, 0], "slope": 3., "center": .6, "maximum": .06}}}
    if suite == "tune":
        for value in fixed_grid:
            key = "fixed_" + format(value, ".6g").replace(".", "p")
            result[key] = {"coefficients": [value] * count, "kind": "fixed", "value": value}
        result.update(tuning_candidates(stats))
        return result
    result["fixed_mean"] = {"coefficients": [float(np.mean(vectors["cags"]))] * count,
                            "kind": "fixed_mean"}
    if suite in {"core", "full"}:
        if best_fixed is None:
            raise ValueError("Core/full requires --selection from held-out tuning.")
        result["fixed_best"] = {"coefficients": [best_fixed] * count, "kind": "fixed", "value": best_fixed}
    if suite in {"allocation", "core", "full"}:
        if not permutations or len(set(permutations)) != len(permutations):
            raise ValueError("Permutation seeds must be nonempty and unique.")
        for perm_seed in permutations:
            order = np.random.default_rng(perm_seed).permutation(count).tolist()
            result[f"permuted_{perm_seed}"] = {"coefficients": [vectors["cags"][i] for i in order],
                                                "kind": "permuted", "permutation_seed": perm_seed,
                                                "permutation_of": "cags",
                                                "source_class_indices": order}
    if suite in {"factors", "full"}:
        for key in ["entropy", "dispersion", "four_factor"]:
            result[key] = {"coefficients": vectors[key], "kind": key}
    if suite in {"core", "full"}:
        result["inherited_mgd_fixed_0p10_window25"] = {
            "coefficients": [.1] * count, "kind": "inherited_mgd_kernel",
            "guidance_steps": 25,
            "note": "Local inherited MGD3 kernel, shared targets/evaluator; published settings not independently verified"}
        result["inherited_mgd_fixed_0p10_stop0"] = {
            "coefficients": [.1] * count, "kind": "inherited_mgd_kernel", "stop_t": 0,
            "note": "Defaults of local sample_mode_guidance.py with --guidance; not a verified reproduction of published numbers"}
    if suite == "cfg_controls":
        mean = float(np.mean(vectors["cags"]))
        for name, policy in [("cfg_anneal", "linear_anneal"), ("cfg_interval", "high_noise_interval")]:
            result[name] = {"coefficients": [mean] * count, "kind": "internal_cfg_control", "cfg_policy": policy}
            result[name + "_cags"] = {"coefficients": vectors["cags"], "kind": "internal_cfg_control",
                                       "cfg_policy": policy}
    return result
