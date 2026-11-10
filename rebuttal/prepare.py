"""Prepare a train/tuning split, deterministic features, and shared mode targets."""
import hashlib
import io
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import ROOT, atomic_json, digest, file_hash, hardware, now, read_json, seed_everything, seed_for, source_manifest

from .image_formats import EXTENSIONS


def center_crop(image, size=256):
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), Image.Resampling.BICUBIC)
    array = np.asarray(image)
    top, left = (array.shape[0] - size) // 2, (array.shape[1] - size) // 2
    return Image.fromarray(array[top:top + size, left:left + size])


def make_split(data_root, classes, fraction, seed, ipc):
    data_root = Path(data_root).resolve()
    if not 0 < fraction < .5:
        raise ValueError("Use a held-out tuning fraction between 0 and 0.5.")
    train, tune = {}, {}
    for cls in classes:
        paths = sorted(p for p in (data_root / "train" / cls).iterdir()
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if len(paths) < ipc + 2:
            raise ValueError(f"Too few source images for {cls}: {len(paths)}")
        rng = np.random.default_rng(seed_for("split", seed, cls))
        order = rng.permutation(len(paths))
        n_tune = max(1, math.floor(len(paths) * fraction))
        if len(paths) - n_tune < ipc:
            raise ValueError(f"Insufficient non-held-out images for IPC={ipc}: {cls}")
        tune_ids = set(order[:n_tune].tolist())
        train[cls] = [str(p.relative_to(data_root)) for i, p in enumerate(paths) if i not in tune_ids]
        tune[cls] = [str(p.relative_to(data_root)) for i, p in enumerate(paths) if i in tune_ids]
    return {"root": str(data_root), "classes": classes, "seed": seed,
            "holdout_fraction": fraction, "train": train, "tune": tune}


class SourceImages(Dataset):
    def __init__(self, root, paths, image_size, flip_seed=None, resnet=False):
        self.root, self.paths = Path(root), paths
        self.image_size, self.flip_seed, self.resnet = image_size, flip_seed, resnet

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from torchvision.transforms import functional as TF
        rel = self.paths[index]
        raw = (self.root / rel).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            image = center_crop(image.convert("RGB"), self.image_size)
            if self.flip_seed is not None and seed_for("feature_flip", self.flip_seed, rel) % 2:
                image = TF.hflip(image)
            tensor = TF.to_tensor(image)
            if self.resnet:
                tensor = TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225])
            else:
                tensor = tensor * 2 - 1
        return tensor, rel, hashlib.sha256(raw).hexdigest()


def parameter_hash(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def profile_from_features(features, classes, k_max, pca_components, fixed_k=None, random_seed=0):
    from ags.class_complexity import ClassComplexityAnalyzer
    analyzer = ClassComplexityAnalyzer(n_clusters_range=(2, k_max), pca_components=pca_components)
    records, centroids = {}, []
    for index, cls in enumerate(classes):
        # PCA's randomized solver must see the same per-class stream across K
        # variants; earlier classes may consume different numbers of RNG draws.
        np.random.seed(seed_for("complexity_class", random_seed, cls))
        x = np.load(features / f"{cls}.npy", mmap_mode="r")
        analyzer.compute_complexity(x, index, fixed_k=fixed_k)
        labels = analyzer.mode_id_per_class[index]
        counts = np.bincount(labels, minlength=analyzer.mode_counts[index])
        proportions = counts[counts > 0] / len(labels)
        entropy = float(-np.sum(proportions * np.log(proportions + 1e-8)))
        k = analyzer.mode_counts[index]
        records[cls] = {"k": int(k), "entropy_normalized": entropy / np.log(k) if k > 1 else 0.0,
                        "dispersion": float(analyzer.intra_variances[index])}
        centroids.append(analyzer.all_centroids[index])
    for index, cls in enumerate(classes):
        distances = [float(np.linalg.norm(centroids[index] - other))
                     for j, other in enumerate(centroids) if j != index]
        ratio = np.mean(distances) / (records[cls]["dispersion"] + 1e-8) if distances else 2.0
        records[cls]["separability"] = float(1 / (1 + np.exp(-np.clip(ratio - 2.0, -80, 80))))
    return {"classes": classes, "k_max": k_max, "fixed_k": fixed_k, "random_seed": random_seed,
            "pca_components": pca_components, "per_class": records,
            "entropy_definition": "H/log(K)", "dispersion_definition": "mean Euclidean distance to centroid",
            "separability_definition": "sigmoid(mean other-centroid distance / dispersion - 2)"}


def prepare(args):
    from diffusers import AutoencoderKL
    from sklearn.cluster import KMeans
    folder = Path(args.workdir).resolve() / "prepared"
    if (folder / "manifest.json").exists():
        raise FileExistsError(f"Prepared cache is immutable: {folder}. Use a new WORKDIR to change it.")
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".prepare.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        classes = [x.strip() for x in Path(args.class_file).read_text().splitlines() if x.strip()][:args.nclass]
        all_classes = (ROOT / "misc/class_indices.txt").read_text().splitlines()
        if len(classes) != args.nclass or len(set(classes)) != len(classes):
            raise ValueError("Class list must contain nclass distinct ImageNet WNIDs.")
        labels = [all_classes.index(c) for c in classes]
        split = make_split(args.data_root, classes, args.holdout_fraction, args.seed, args.ipc)
        atomic_json(folder / "split.json", split)
        seed_everything(args.seed, args.deterministic)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Preparation requires an accessible GPU; use --device cpu only for small tests.")
        vae = AutoencoderKL.from_pretrained(args.vae, local_files_only=args.local_files_only).to(device).eval()
        vae_hash = parameter_hash(vae)
        features = folder / "features"
        features.mkdir(exist_ok=True)
        source_hashes = {}
        targets = {}
        for cls in classes:
            data = SourceImages(split["root"], split["train"][cls], args.image_size,
                                args.seed if args.feature_flip else None)
            loader = DataLoader(data, batch_size=args.feature_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
            chunks = []
            with torch.no_grad():
                for images, paths, hashes in loader:
                    latent = vae.encode(images.to(device)).latent_dist.mean * .18215
                    chunks.append(latent.flatten(1).cpu().numpy())
                    source_hashes.update(zip(paths, hashes))
            x = np.concatenate(chunks).astype(np.float32)
            np.save(features / f"{cls}.npy", x)
            targets[cls] = KMeans(n_clusters=args.ipc, random_state=0, n_init="auto").fit(x).cluster_centers_
            print(f"Prepared {cls}: {len(x)} feature vectors; held out {len(split['tune'][cls])}", flush=True)
        np.savez(folder / "mode_targets.npz", **targets)
        atomic_json(folder / "source_image_hashes.json", source_hashes)
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
        profile_dir = folder / "profiles"
        profile_dir.mkdir(exist_ok=True)
        profiles = [("vae_auto", features, None, args.pca_components)]
        profiles += [(f"vae_k{k}", features, k, args.pca_components) for k in args.k_values]
        profiles += [(f"vae_pca{d}", features, None, d) for d in args.pca_variants if d != args.pca_components]
        resnet_hash = None
        if args.resnet_checkpoint:
            from torchvision.models import resnet18
            network = resnet18(weights=None)
            state = torch.load(args.resnet_checkpoint, map_location="cpu")
            network.load_state_dict(state, strict=True)
            network.fc = torch.nn.Identity()
            network.to(device).eval()
            resnet_hash = file_hash(args.resnet_checkpoint)
            alt = folder / "features_resnet18"
            alt.mkdir(exist_ok=True)
            for cls in classes:
                loader = DataLoader(SourceImages(split["root"], split["train"][cls], 224,
                                                args.seed if args.feature_flip else None, resnet=True),
                                    batch_size=args.feature_batch_size, num_workers=args.num_workers)
                with torch.no_grad():
                    x = np.concatenate([network(images.to(device)).cpu().numpy() for images, _, _ in loader])
                np.save(alt / f"{cls}.npy", x.astype(np.float32))
            del network
            profiles.append(("resnet18_auto", alt, None, args.pca_components))
        for name, path, k, dimension in profiles:
            seed_everything(seed_for("profile", args.seed, name), args.deterministic)
            stats = profile_from_features(path, classes, args.k_max, dimension, k, random_seed=args.seed)
            atomic_json(profile_dir / f"{name}.json", stats)
        hashes = {str(p.relative_to(folder)): file_hash(p) for p in folder.rglob("*")
                  if p.is_file() and p.name not in {"manifest.json", ".prepare.lock"}}
        config = {k: v for k, v in vars(args).items() if k not in {"func"}}
        atomic_json(folder / "manifest.json", {"schema": 1, "created": now(), "config": config,
                    "classes": classes, "imagenet_labels": labels, "ipc": args.ipc,
                    "image_size": args.image_size, "vae": args.vae, "vae_parameter_sha256": vae_hash,
                    "resnet_checkpoint_sha256": resnet_hash, "files": hashes,
                    "source": source_manifest(), "hardware": hardware(),
                    "profiles": [name for name, _, _, _ in profiles], "split_hash": digest(split)})
        print(f"Immutable prepared cache: {folder}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
