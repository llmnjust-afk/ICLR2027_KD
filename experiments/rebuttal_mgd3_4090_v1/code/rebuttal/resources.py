"""Locate local resources and download missing, pinned model/data artifacts.

ImageNet labels are resolved from WNIDs, dataset metadata, or original filenames.
The order of misc/class100.txt is NEVER assumed to be the dataset's label order.
"""
import argparse
import fcntl
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from .common import ROOT, atomic_json, file_hash, now, read_json
from .transfers import hub_download
from .image_formats import EXTENSIONS, inspect_image

WNID = re.compile(r"(?<![A-Za-z0-9])n\d{8}(?!\d)")


def configured(name):
    value = os.environ.get(name, "")
    return value if value and not value.startswith("/path/to/") else None


def dataset_check(root, classes, ipc=10, report=None, allow_unfinished=False):
    root = Path(root)
    report = Path(report) if report else root / "val"
    counts, missing = {}, []
    for cls in classes:
        counts[cls] = {}
        for split, base in [("train", root / "train"), ("val", report)]:
            try:
                n = sum(p.is_file() and p.suffix.lower() in EXTENSIONS for p in (base / cls).iterdir())
            except OSError:
                n = 0
            counts[cls][split] = n
            if n < (ipc + 2 if split == "train" else 1):
                missing.append(f"{split}/{cls}")
    completed = root / "resource_manifest.json"
    if (root / "download_source.json").exists() and not completed.exists() and not allow_unfinished:
        missing.append("managed download has not finished all shards")
    if completed.exists():
        expected = read_json(completed).get("counts", {})
        if expected and any(expected.get(cls) != counts[cls] for cls in classes):
            missing.append("counts differ from completed download manifest")
    return {"root": str(root), "report": str(report), "valid": not missing,
            "missing": missing, "counts": counts}


def has_vae(path):
    try:
        path = Path(path)
        config = read_json(path / "config.json")
        return config.get("_class_name") == "AutoencoderKL" and any(
            p.is_file() and p.stat().st_size > 1_000_000 for p in path.glob("diffusion_pytorch_model.*")
            if p.suffix in {".bin", ".safetensors"})
    except (OSError, ValueError):
        return False


def vae_candidates(resource_root):
    candidates = [configured("VAE_PATH"), resource_root / "models/sd-vae-ft-mse",
                  ROOT / "pretrained_models/sd-vae-ft-mse", ROOT / "sd-vae-ft-mse"]
    caches = [Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
              resource_root / "hf_cache"]
    for cache in caches:
        candidates.extend((cache / "models--stabilityai--sd-vae-ft-mse/snapshots").glob("*"))
    return [Path(p) for p in candidates if p]


def dit_candidates(resource_root):
    return [Path(p) for p in [configured("DIT_CKPT"), ROOT / "pretrained_models/DiT-XL-2-256x256.pt",
            resource_root / "models/DiT-XL-2-256x256.pt",
            ROOT.parent / "MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt"] if p]


def data_candidates(resource_root):
    candidates = [configured("DATA_ROOT"), resource_root / "datasets/imagenet100",
                  ROOT / "data/imagenet100", ROOT / "data/imagenet", Path("/root/data/imagenet100"),
                  Path("/data/datasets/ImageNet"), Path("/ssd_data/imagenet")]
    try:
        projects = [p for p in ROOT.parent.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError:
        projects = []
    for project in projects:
        candidates.extend(project / rel for rel in ["data/datasets/ImageNet", "data/imagenet100",
                                                   "data/imagenet", "datasets/imagenet"])
    return list(dict.fromkeys(Path(p) for p in candidates if p))


def inventory(resource_root, classes, ipc):
    dits = []
    for path in dit_candidates(resource_root):
        try:
            if path.is_file() and path.stat().st_size > 1_000_000_000:
                dits.append(str(path.resolve()))
        except OSError:
            pass
    vaes = [str(p.resolve()) for p in vae_candidates(resource_root) if has_vae(p)]
    data, rejected = [], []
    for candidate in data_candidates(resource_root):
        try:
            if not (candidate / "train").is_dir():
                continue
        except OSError:
            continue
        explicit = configured("DATA_ROOT")
        check = dataset_check(candidate, classes, ipc,
                              configured("REPORT_DIR") if explicit and candidate == Path(explicit) else None)
        if check["valid"]:
            data.append(check)
        else:
            rejected.append({"root": str(candidate), "missing_splits_or_classes": len(check["missing"])})
    return {"created": now(), "dit": list(dict.fromkeys(dits)), "exact_or_configured_vae": vaes,
            "datasets": data, "incomplete_dataset_candidates": rejected}


def link_local(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != Path(source).resolve():
            raise ValueError(f"Refusing to replace an existing resource: {destination}")
    else:
        destination.symlink_to(Path(source).resolve(), target_is_directory=Path(source).is_dir())
    return str(destination)


def ensure_dit(resource_root, found):
    target = ROOT / "pretrained_models/DiT-XL-2-256x256.pt"
    explicit = configured("DIT_CKPT")
    if explicit and Path(explicit).is_file() and str(Path(explicit).resolve()) in found:
        return str(Path(explicit).resolve())
    if found:
        if target.exists() and target.resolve() == Path(found[0]).resolve():
            return str(target)
        return link_local(found[0], target)
    import requests
    target.parent.mkdir(exist_ok=True)
    partial = target.with_suffix(".pt.partial")
    url = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
    offset = partial.stat().st_size if partial.exists() else 0
    with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        mode = "ab" if response.status_code == 206 else "wb"
        with partial.open(mode) as output:
            for chunk in response.iter_content(8 * 1024 * 1024):
                output.write(chunk)
    if partial.stat().st_size < 1_000_000_000:
        raise ValueError("Downloaded DiT file is unexpectedly small; refusing to use it.")
    # Check that a truncated transfer is not promoted to a completed checkpoint.
    import torch
    state = torch.load(partial, map_location="cpu")
    if "x_embedder.proj.weight" not in state:
        raise ValueError("Unexpected pretrained DiT checkpoint format.")
    del state
    partial.replace(target)
    return str(target)


def ensure_vae(resource_root, found):
    if found:
        return found[0]
    from huggingface_hub import HfApi, hf_hub_download
    repo = "stabilityai/sd-vae-ft-mse"
    info = HfApi(token=False).model_info(repo, timeout=20)
    names = {p.rfilename for p in info.siblings}
    weights = "diffusion_pytorch_model.safetensors" if "diffusion_pytorch_model.safetensors" in names else "diffusion_pytorch_model.bin"
    print(f"Downloading exact original VAE {repo}@{info.sha}", flush=True)
    cache = resource_root / "hf_cache"
    config = hf_hub_download(repo, "config.json", revision=info.sha, cache_dir=str(cache), token=False)
    weight_file = hub_download(repo, weights, revision=info.sha, cache_dir=cache)
    snapshot = Path(config).parent
    # curl stores a verified Hub blob; materialize the matching snapshot entry.
    link_local(weight_file, snapshot / weights)
    if not has_vae(snapshot):
        raise ValueError("VAE download is incomplete or has an unexpected architecture.")
    target = resource_root / "models/sd-vae-ft-mse"
    link_local(snapshot, target)
    atomic_json(resource_root / "vae_source.json", {"repo": repo, "revision": info.sha,
                "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "files": {p.name: file_hash(p) for p in Path(snapshot).iterdir() if p.is_file()}, "created": now()})
    return str(target)


def require_arrow(resource_root):
    try:
        import pyarrow.parquet
        return
    except ImportError:
        pass
    compatible = []
    for wheel in (resource_root / "wheels").glob("pyarrow-*.whl"):
        if f"cp{sys.version_info.major}{sys.version_info.minor}-" in wheel.name:
            compatible.append(str(wheel))
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir"]
    if compatible:
        command += ["--no-index", "--no-deps", compatible[0]]
    else:
        command += ["--only-binary=:all:", "pyarrow==17.0.0"]
    subprocess.run(command, check=True)


def normalize_name(value):
    return re.sub(r"\s+", " ", str(value).replace("_", " ").strip().lower())


def imagenet_aliases():
    ids = (ROOT / "misc/class_indices.txt").read_text().splitlines()
    names = (ROOT / "misc/class_names.txt").read_text().splitlines()
    out = defaultdict(set)
    for wnid, name in zip(ids, names):
        out[normalize_name(name)].add(wnid)
        for alias in name.split(","):
            out[normalize_name(alias)].add(wnid)
    return out


def name_to_wnid(value, aliases):
    matches = set(WNID.findall(str(value)))
    if len(matches) == 1:
        return next(iter(matches))
    candidates = aliases.get(normalize_name(value), set())
    return next(iter(candidates)) if len(candidates) == 1 else None


def metadata_names(metadata):
    features = metadata.get("info", metadata).get("features", {})
    if isinstance(features, list):
        feature = next((f.get("dtype", {}) for f in features if f.get("name") == "label"), {})
    else:
        feature = features.get("label", {})
    if not isinstance(feature, dict):
        return None
    return feature.get("names") or feature.get("class_label", {}).get("names")


def resolve_row_wnid(row, names, inferred, aliases):
    candidates = set()
    for key in ["wnid", "synset", "class_name", "class_id"]:
        if isinstance(row.get(key), str):
            value = name_to_wnid(row[key], aliases)
            if value:
                candidates.add(value)
    image = row.get("image")
    if isinstance(image, dict) and image.get("path"):
        # Only use WNIDs in paths; ambiguous human directory names are not labels.
        matches = set(WNID.findall(image["path"]))
        if len(matches) == 1:
            candidates.update(matches)
    label = row.get("label")
    if isinstance(label, str):
        wnid = name_to_wnid(label, aliases)
        if wnid:
            candidates.add(wnid)
    elif label is not None and names:
        name = names.get(str(label), names.get(label)) if isinstance(names, dict) else names[int(label)]
        wnid = name_to_wnid(name, aliases)
        if wnid:
            candidates.add(wnid)
    if str(label) in inferred:
        candidates.add(inferred[str(label)])
    if len(candidates) != 1:
        raise ValueError(f"Cannot uniquely verify ImageNet WNID for label={label}: {sorted(candidates)}. "
                         "Refusing to assume that dataset label order equals misc/class100.txt order.")
    wnid = next(iter(candidates))
    if label is not None:
        inferred[str(label)] = wnid
    return wnid


def ensure_dataset(resource_root, found, classes, ipc):
    if found:
        return found[0]["root"], found[0]["report"]
    require_arrow(resource_root)
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    repo = os.environ.get("REBUTTAL_DATASET_REPO", "clane9/imagenet-100")
    target = resource_root / "datasets/imagenet100"
    target.mkdir(parents=True, exist_ok=True)
    state_file = target / "download_source.json"
    api = HfApi(token=False)
    old = read_json(state_file) if state_file.exists() else None
    dataset_info = api.dataset_info(repo, revision=old["revision"] if old else None, timeout=20)
    revision = old["revision"] if old else dataset_info.sha
    if old and old["repo"] != repo:
        raise ValueError("Existing partial download belongs to another dataset repository.")
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    shards = {split: sorted(f for f in files if f.endswith(".parquet") and
                            re.search(r"(?:^|/)(?:" + ("train" if split == "train" else "validation|val|test") + r")(?:[-_./]|$)", f))
              for split in ["train", "val"]}
    if not all(shards.values()):
        raise ValueError(f"Dataset does not expose expected train/validation parquet files: {repo}")
    atomic_json(state_file, {"repo": repo, "revision": revision, "created": old.get("created", now()) if old else now()})
    markers = target / "download_records"
    markers.mkdir(exist_ok=True)
    aliases, inferred = imagenet_aliases(), {}
    # Load explicitly published ClassLabel metadata when present.
    names = None
    card = getattr(dataset_info, "card_data", None) or getattr(dataset_info, "cardData", None)
    if card is not None:
        card = card.to_dict() if hasattr(card, "to_dict") else dict(card)
        candidate = card.get("dataset_info", {})
        for value in candidate if isinstance(candidate, list) else [candidate]:
            if isinstance(value, dict):
                names = metadata_names(value)
                if names:
                    break
    if "dataset_info.json" in files:
        metadata_file = hf_hub_download(repo, "dataset_info.json", repo_type="dataset", revision=revision,
                                        cache_dir=str(resource_root / "hf_cache"), token=False)
        info = read_json(metadata_file)
        file_names = metadata_names(info)
        if not file_names:
            for value in info.values():
                if isinstance(value, dict):
                    file_names = metadata_names(value)
                    if file_names:
                        break
        names = file_names or names
    for split in ["train", "val"]:
        for shard in shards[split]:
            marker = markers / (Path(shard).stem + ".done.json")
            if marker.exists():
                saved = read_json(marker)
                if saved["revision"] != revision:
                    raise ValueError("Shard marker has another dataset revision.")
                inferred.update(saved["label_to_wnid"])
                record = markers / (Path(shard).stem + ".jsonl")
                if file_hash(record) != saved["record_sha256"]:
                    raise ValueError(f"Corrupted extraction record: {record}")
                if all((target / json.loads(line)["output"]).is_file() for line in record.read_text().splitlines()):
                    continue
                marker.rename(marker.with_name(marker.name + f".incomplete_{time.time_ns()}"))
            print(f"Downloading/extracting {repo}@{revision}: {shard}", flush=True)
            local = hub_download(repo, shard, repo_type="dataset", revision=revision,
                                 cache_dir=str(resource_root / "hf_cache"), token=False)
            parquet = pq.ParquetFile(local)
            meta = parquet.schema_arrow.metadata or {}
            shard_names = metadata_names(json.loads(meta[b"huggingface"])) if b"huggingface" in meta else None
            row_index, written = 0, 0
            record = markers / (Path(shard).stem + ".jsonl")
            with record.open("w") as log:
                for batch in parquet.iter_batches(batch_size=64):
                    columns = batch.to_pydict()
                    for offset in range(batch.num_rows):
                        row = {key: values[offset] for key, values in columns.items()}
                        wnid = resolve_row_wnid(row, shard_names or names, inferred, aliases)
                        index = row_index
                        row_index += 1
                        if wnid not in classes:
                            continue
                        value = row["image"]
                        raw = value.get("bytes") if isinstance(value, dict) else value
                        if not isinstance(raw, bytes):
                            raise ValueError("Expected embedded image bytes in the parquet dataset.")
                        decoded = inspect_image(io.BytesIO(raw), context=f"shard={shard}, row={index}, wnid={wnid}")
                        ext = decoded["extension"]
                        destination = target / split / wnid / f"{Path(shard).stem}_{index:08d}{ext}"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        checksum = __import__("hashlib").sha256(raw).hexdigest()
                        if destination.exists():
                            if file_hash(destination) != checksum:
                                raise ValueError(f"Existing extracted image differs from source: {destination}")
                        else:
                            temporary = destination.with_suffix(destination.suffix + ".partial")
                            temporary.write_bytes(raw)
                            temporary.replace(destination)
                        log.write(json.dumps({"row": index, "wnid": wnid, "label": row.get("label"),
                                              "output": str(destination.relative_to(target)), "sha256": checksum}) + "\n")
                        written += 1
            atomic_json(marker, {"repo": repo, "revision": revision, "shard": shard, "rows": row_index,
                                 "written": written, "label_to_wnid": inferred,
                                 "record_sha256": file_hash(record), "completed": now()})
    check = dataset_check(target, classes, ipc, allow_unfinished=True)
    if not check["valid"]:
        raise ValueError(f"Downloaded dataset is missing required classes/images: {check['missing'][:10]}")
    if any(value["val"] != 50 for value in check["counts"].values()):
        raise ValueError("Downloaded validation split is not the expected 50 images per ImageNet class.")
    atomic_json(target / "resource_manifest.json", {**check, "repo": repo, "revision": revision,
                "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "label_to_wnid": inferred, "created": now(), "label_policy": "verified WNIDs; never positional class-file mapping"})
    return str(target), str(target / "val")


def write_config(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = ROOT / "backups" / f"resource_config_{time.time_ns()}.env"
        backup.parent.mkdir(exist_ok=True)
        backup.write_bytes(path.read_bytes())
    text = path.read_text() if path.exists() else "# Local resources prepared automatically\n"
    text += "\n# Verified resource paths\n" + "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items())
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "ensure"])
    parser.add_argument("--resource-root", default=str(ROOT / "resources"))
    parser.add_argument("--config", default=str(ROOT / "rebuttal.local.env"))
    parser.add_argument("--class-file", default=str(ROOT / "misc/class100.txt"))
    parser.add_argument("--nclass", type=int, default=100)
    parser.add_argument("--ipc", type=int, default=10)
    args = parser.parse_args()
    resource_root = Path(args.resource_root).resolve()
    resource_root.mkdir(parents=True, exist_ok=True)
    lock = (resource_root / ".resource_preparation.lock").open("a+")
    if args.command == "ensure":
        print("Checking resource preparation lock...", flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    classes = [line.strip() for line in Path(args.class_file).read_text().splitlines() if line.strip()][:args.nclass]
    if len(classes) != args.nclass or len(set(classes)) != len(classes):
        raise ValueError("Expected a complete, unique class list.")
    found = inventory(resource_root, classes, args.ipc)
    atomic_json(resource_root / "inventory.json", found)
    if args.command == "inspect":
        print(json.dumps(found, indent=2))
        return
    print("Preparing resources: reuse local files first; download missing exact artifacts.", flush=True)
    phase = "dit"
    try:
        dit = ensure_dit(resource_root, found["dit"])
        print(f"DiT ready: {dit}", flush=True)
        phase = "vae"
        vae = ensure_vae(resource_root, found["exact_or_configured_vae"])
        print(f"VAE ready: {vae}", flush=True)
        phase = "dataset"
        data, report = ensure_dataset(resource_root, found["datasets"], classes, args.ipc)
    except Exception as error:
        from requests.exceptions import RequestException
        exit_code = 75 if isinstance(error, RequestException) else 1
        atomic_json(resource_root / "last_error.json", {"created": now(), "phase": phase,
                    "exception": type(error).__name__, "message": str(error), "exit_code": exit_code,
                    "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co")})
        print(f"资源尚未齐备：阶段={phase}，错误类型={type(error).__name__}。", file=sys.stderr)
        print(str(error), file=sys.stderr)
        print(f"详细原因：{resource_root / 'last_error.json'}。缺失资源下载需要运行节点能访问 Hugging Face。", file=sys.stderr)
        raise SystemExit(exit_code)
    values = {"DATA_ROOT": data, "REPORT_DIR": report, "DIT_CKPT": dit, "VAE_PATH": vae,
              "LOCAL_ONLY": "1"}
    write_config(args.config, values)
    atomic_json(resource_root / "resources_ready.json", {"created": now(), "paths": values,
                "classes": classes, "class_file_sha256": file_hash(args.class_file)})
    (resource_root / "last_error.json").unlink(missing_ok=True)
    print("Resources ready; local configuration saved.", flush=True)


if __name__ == "__main__":
    main()
