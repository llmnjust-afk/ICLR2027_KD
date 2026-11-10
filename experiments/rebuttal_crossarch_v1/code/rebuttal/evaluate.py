"""Classifier evaluation with fixed-final or tuning-selected checkpoints.

Final reporting images are never used for checkpoint or hyperparameter selection.
The augmentation/model helpers intentionally reuse the existing v3 evaluator.
"""
import gc
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from quick_eval_v3 import Lighting, create_model, load_images_to_tensor, rand_bbox
from .common import atomic_json, file_hash, hardware, now, read_json, seed_everything
from .image_formats import EXTENSIONS


class EvaluationImages(Dataset):
    def __init__(self, root, paths, size):
        self.root, self.paths = Path(root), paths
        self.transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                                             transforms.ToTensor(), transforms.Normalize(
                                                 [.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rel, label = self.paths[index]
        with Image.open(self.root / rel) as image:
            return self.transform(image.convert("RGB")), label


def evaluation_paths(prepared, report_dir, split_name):
    split = read_json(prepared / "split.json")
    if split_name == "tune":
        return split["root"], [(p, i) for i, cls in enumerate(split["classes"]) for p in split["tune"][cls]]
    root = Path(report_dir).resolve()
    paths = []
    for label, cls in enumerate(split["classes"]):
        files = sorted(p for p in (root / cls).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No final-report images for class {cls} in {root}")
        paths.extend((str(p.relative_to(root)), label) for p in files)
    # A user must not accidentally point the reporting directory at the held-out training split.
    training_root = Path(split["root"]) / "train"
    if root == training_root or root.is_relative_to(training_root):
        raise ValueError("--report-dir must be the original validation directory, not training images.")
    return str(root), paths


def image_manifest(root, paths):
    return [{"path": rel, "label": label, "bytes": (Path(root) / rel).stat().st_size,
             "mtime_ns": (Path(root) / rel).stat().st_mtime_ns} for rel, label in paths]


def loader(root, paths, size, batch_size, workers):
    return DataLoader(EvaluationImages(root, paths, size), batch_size=batch_size,
                      shuffle=False, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


@torch.no_grad()
def measure(model, batches, device, count):
    model.eval()
    totals = torch.zeros(count, dtype=torch.long, device=device)
    corrects = torch.zeros_like(totals)
    top5, correct, n = 0, 0, 0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite classifier predictions; refusing to report an accuracy.")
        hit = logits.argmax(1).eq(labels)
        correct += int(hit.sum())
        top5 += int(logits.topk(min(5, count), 1).indices.eq(labels[:, None]).any(1).sum())
        n += len(labels)
        totals += torch.bincount(labels, minlength=count)
        corrects += torch.bincount(labels[hit], minlength=count)
    if n == 0 or bool((totals == 0).any()):
        raise ValueError("Every selected class must have evaluation images.")
    return {"top1": 100 * correct / n, "top5": 100 * top5 / n, "n": n,
            "per_class_correct": corrects.cpu().tolist(), "per_class_total": totals.cpu().tolist()}


def train_one(train_dir, classes, cfg, train_seed, architecture, evaluation_loader,
              tuning_loader, output, provenance):
    device = torch.device(cfg["device"])
    seed_everything(train_seed, cfg["deterministic"])
    torch.set_num_threads(cfg["threads"])
    images, labels = load_images_to_tensor(str(train_dir), classes, cfg["classifier_size"])
    counts = torch.bincount(labels, minlength=len(classes))
    if not torch.all(counts == cfg["ipc"]):
        raise ValueError(f"Expected exactly IPC={cfg['ipc']} images/class; got {counts.tolist()}")
    images, labels = images.to(device), labels.to(device)
    arch, depth = {"convnet6": ("convnet", 6), "resnet18": ("resnet", 18),
                   "resnetap10": ("resnet_ap", 10)}[architecture]
    model = create_model(arch, len(classes), depth=depth, norm_type="instance",
                         img_size=cfg["classifier_size"]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=.9,
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [2 * cfg["epochs"] // 3, 5 * cfg["epochs"] // 6], gamma=.2)
    crop = transforms.RandomResizedCrop(cfg["classifier_size"], scale=(.5, 1), antialias=True)
    flip = transforms.RandomHorizontalFlip()
    color = transforms.ColorJitter(brightness=.4, contrast=.4, saturation=.4)
    lighting = Lighting(.1)
    use_amp = cfg["precision"] == "amp"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    started, best_tune, selected_epoch, best_state = time.time(), -1.0, cfg["epochs"], None
    history = output.with_suffix(".history.jsonl")
    history.parent.mkdir(parents=True, exist_ok=True)
    # A failed run restarts this training seed, never appends to a previous partial trajectory.
    with history.open("w") as stream:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            order = torch.randperm(len(images), device=device)
            loss_sum = 0.0
            for start in range(0, len(images), cfg["batch_size"]):
                idx = order[start:start + cfg["batch_size"]]
                batch, target = images[idx].clone(), labels[idx]
                batch = TF.normalize(lighting(color(flip(crop(batch)))), [.485, .456, .406], [.229, .224, .225])
                lam = np.random.beta(1, 1)
                other = torch.randperm(len(batch), device=device)
                x1, y1, x2, y2 = rand_bbox(batch.size(), lam)
                batch[:, :, x1:x2, y1:y2] = batch[other, :, x1:x2, y1:y2]
                ratio = 1 - (x2 - x1) * (y2 - y1) / (batch.size(-1) * batch.size(-2))
                optimizer.zero_grad(set_to_none=True)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else nullcontext()
                with autocast:
                    logits = model(batch)
                    loss = torch.nn.functional.cross_entropy(logits, target) * ratio
                    loss += torch.nn.functional.cross_entropy(logits, target[other]) * (1 - ratio)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(batch)
            scheduler.step()
            tune = None
            if cfg["checkpoint_rule"] == "tune_best" and (epoch % cfg["eval_interval"] == 0 or epoch == cfg["epochs"]):
                tune = measure(model, tuning_loader, device, len(classes))["top1"]
                if tune > best_tune:
                    best_tune, selected_epoch = tune, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % cfg["eval_interval"] == 0 or epoch == cfg["epochs"]:
                import json
                row = {"epoch": epoch, "loss": loss_sum / len(images), "tuning_top1": tune,
                       "elapsed_s": time.time() - started}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"{output.stem} seed={train_seed} epoch={epoch}/{cfg['epochs']} loss={row['loss']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    # The reporting loader is visited once, after the checkpoint decision is frozen.
    metrics = measure(model, evaluation_loader, device, len(classes))
    checkpoint = output.with_suffix(".pt")
    if cfg["save_checkpoints"]:
        temp = checkpoint.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "selected_epoch": selected_epoch,
                    "architecture": architecture, "classes": classes, "provenance": provenance}, temp)
        temp.replace(checkpoint)
    result = {**provenance, "created": now(), "architecture": architecture, "train_seed": train_seed,
              "metrics": metrics, "selected_epoch": selected_epoch, "checkpoint_rule": cfg["checkpoint_rule"],
              "best_tuning_top1": best_tune if best_state is not None else None,
              "elapsed_s": time.time() - started, "hardware": hardware(),
              "checkpoint_sha256": file_hash(checkpoint) if cfg["save_checkpoints"] else None}
    atomic_json(output, result)
    del model, optimizer, scheduler, images, labels, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
