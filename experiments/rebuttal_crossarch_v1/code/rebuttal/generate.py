"""Generation using the existing sampler and dedicated paired RNG streams."""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from ags.ags_sampler import AGSSampler
from diffusion import create_diffusion
from models import DiT_models
from .common import atomic_json, digest, file_hash, hardware, now, read_json, seed_everything, seed_for
from .prepare import parameter_hash
from .checkpoints import load_compatible


def cfg_policy(name):
    if name == "constant":
        return None
    if name == "linear_anneal":
        return lambda step, total, base: 1 + (base - 1) * step / max(total - 1, 1)
    if name == "high_noise_interval":
        return lambda step, total, base: base if step >= total // 2 else 1.0
    raise ValueError(f"Unknown CFG policy: {name}")


def dataset_dir(plan_dir, generation_seed, condition):
    return Path(plan_dir) / "datasets" / f"g{generation_seed}" / condition


def verify_dataset(folder, expected=None):
    manifest = read_json(Path(folder) / "manifest.json")
    if expected is not None and manifest["configuration_hash"] != expected:
        raise ValueError(f"Dataset configuration mismatch: {folder}")
    for rel, checksum in manifest["images"].items():
        if file_hash(Path(folder) / rel) != checksum:
            raise ValueError(f"Generated image hash mismatch: {folder}/{rel}")
    actual = {str(p.relative_to(folder)) for p in Path(folder).rglob("*.png")}
    if actual != set(manifest["images"]):
        raise ValueError(f"Extra or missing generated PNG images: {folder}")
    return manifest


class GenerationEngine:
    def __init__(self, plan):
        from diffusers import AutoencoderKL
        cfg = plan["protocol"]
        self.device = torch.device(cfg["device"])
        seed_everything(0, cfg["deterministic"])
        self.model = DiT_models["DiT-XL/2"](input_size=cfg["image_size"] // 8, num_classes=1000)
        state = torch.load(cfg["checkpoint"], map_location="cpu")
        # This sampler never supplies m; absent gamma adapters must remain exactly one.
        self.checkpoint_keys = load_compatible(self.model, state, mode_embedding_unused=True)
        del state
        self.model.to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(cfg["vae"], local_files_only=cfg["local_files_only"]).to(self.device).eval()
        if parameter_hash(self.vae) != plan["prepared_manifest"]["vae_parameter_sha256"]:
            raise ValueError("Generation VAE differs from the feature-extraction VAE.")
        self.diffusion = create_diffusion(str(cfg["steps"]))
        self.targets = np.load(Path(cfg["prepared"]) / "mode_targets.npz")

    @torch.no_grad()
    def generate(self, plan, plan_dir, generation_seed, name, condition):
        cfg = plan["protocol"]
        folder = dataset_dir(plan_dir, generation_seed, name)
        identity = {"plan_hash": plan["plan_hash"], "generation_seed": generation_seed,
                    "condition": name, "specification": condition}
        config_hash = digest(identity)
        if (folder / "manifest.json").exists():
            return verify_dataset(folder, config_hash)
        folder.mkdir(parents=True, exist_ok=True)
        pending = folder / "generation_pending.json"
        if pending.exists() and read_json(pending)["configuration_hash"] != config_hash:
            raise ValueError(f"Refusing to overwrite another experiment: {folder}")
        atomic_json(pending, {**identity, "configuration_hash": config_hash, "started": now()})
        sampler = AGSSampler(self.model, self.vae, self.diffusion, device=str(self.device),
                             cfg_scale=cfg["cfg_scale"], num_sampling_steps=cfg["steps"],
                             latent_size=cfg["image_size"] // 8, use_cags=False,
                             use_iast=False, use_tags=False, guidance_window="high_noise")
        t_stop = condition.get("stop_t", cfg["steps"] - 1 - condition.get("guidance_steps", cfg["guidance_steps"]))
        # No analysis calls and no global RNG are used after these coefficients are fixed.
        sampler.per_class_params = {
            label: {"guidance_strength": float(condition["coefficients"][index]), "t_stop": t_stop}
            for index, label in enumerate(plan["prepared_manifest"]["imagenet_labels"])}
        started = time.time()
        images, noise_seeds = {}, {}
        for index, (cls, label) in enumerate(zip(plan["classes"], plan["prepared_manifest"]["imagenet_labels"])):
            (folder / cls).mkdir(exist_ok=True)
            for start in range(0, cfg["ipc"], cfg["generation_batch_size"]):
                count = min(cfg["generation_batch_size"], cfg["ipc"] - start)
                # The stream excludes method/condition. All methods use identical initial
                # and per-step noises for the same class/chunk, including zero-guidance branches.
                noise_seed = seed_for("generation", generation_seed, label, start)
                noise_seeds[f"{cls}/{start}"] = noise_seed
                generator = torch.Generator(device=self.device).manual_seed(noise_seed)
                z = torch.randn((count, 4, cfg["image_size"] // 8, cfg["image_size"] // 8),
                                generator=generator, device=self.device)
                z = torch.cat([z, z])
                y = torch.tensor([label] * count + [1000] * count, device=self.device)
                modes = torch.tensor(self.targets[cls][start:start + count], device=self.device).reshape(
                    count, 4, cfg["image_size"] // 8, cfg["image_size"] // 8)
                latent = sampler.sample_with_ags_guidance(
                    z, label, modes, {"y": y, "cfg_scale": cfg["cfg_scale"]},
                    generator=generator, cfg_schedule=cfg_policy(condition.get("cfg_policy", "constant")))
                decoded = self.vae.decode(latent.chunk(2)[0] / .18215).sample
                if not bool(torch.isfinite(decoded).all()):
                    raise FloatingPointError(f"Non-finite generated pixels: {name}, {cls}, sample {start}")
                for offset, image in enumerate(decoded):
                    rel = f"{cls}/{start + offset:04d}.png"
                    path = folder / rel
                    temp = path.with_suffix(".partial.png")
                    save_image(image, str(temp), normalize=True, value_range=(-1, 1))
                    temp.replace(path)
                    images[rel] = file_hash(path)
            print(f"{name} g={generation_seed}: class {index + 1}/{len(plan['classes'])} {cls}", flush=True)
        manifest = {**identity, "configuration_hash": config_hash, "created": now(),
                    "images": images, "noise_seeds": noise_seeds, "noise_pairing": "dedicated stream per generation/class/chunk",
                    "hardware": hardware(), "checkpoint_keys": self.checkpoint_keys,
                    "elapsed_s": time.time() - started, "cfg_scale": cfg["cfg_scale"],
                    "ipc": cfg["ipc"], "classes": plan["classes"],
                    "mode_guidance_by_class": dict(zip(plan["classes"], condition["coefficients"])),
                    "active_guidance_steps": [i for i in range(cfg["steps"]) if i > t_stop]}
        atomic_json(folder / "manifest.json", manifest)
        pending.unlink(missing_ok=True)
        return manifest

    def close(self):
        self.targets.close()
        del self.model, self.vae
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
