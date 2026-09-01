import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import argparse
import time


class ConvNet(nn.Module):
    def __init__(self, channel=128, num_classes=10, depth=6, norm="instance"):
        super().__init__()
        self.features = nn.ModuleList()
        in_ch = 3
        for i in range(depth):
            out_ch = channel
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            if norm == "instance":
                norm_layer = nn.InstanceNorm2d(out_ch)
            elif norm == "batch":
                norm_layer = nn.BatchNorm2d(out_ch)
            else:
                norm_layer = nn.Identity()
            block = nn.Sequential(
                conv, norm_layer, nn.ReLU(inplace=True), nn.AvgPool2d(2),
            )
            self.features.append(block)
            in_ch = out_ch
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def load_images_to_tensor(data_dir, class_names, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        for f in files:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            images.append(img)
            labels.append(idx)
    return torch.stack(images), torch.tensor(labels)


def load_val_data(val_dir, class_names, img_size=224, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(glob.glob(os.path.join(cls_dir, "*.JPEG")) + glob.glob(os.path.join(cls_dir, "*.jpg")))
        for f in files:
            img = Image.open(f).convert("RGB")
            img = transform(img)
            images.append(img)
            labels.append(idx)
    return torch.stack(images), torch.tensor(labels)


def evaluate_fast(train_dir, val_dir, class_names, num_classes, device,
                  img_size=224, epochs=1000, batch_size=64, seed=0, depth=6):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading training images from {train_dir}...")
    train_images, train_labels = load_images_to_tensor(train_dir, class_names, img_size)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    print(f"  Loaded {len(train_images)} training images")

    print(f"  Loading validation images from {val_dir}...")
    val_images, val_labels = load_val_data(val_dir, class_names, img_size)
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)
    print(f"  Loaded {len(val_images)} validation images")

    model = ConvNet(num_classes=num_classes, depth=depth).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_images)
    rrc = transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            imgs = train_images[idx].clone()
            tgt = train_labels[idx]

            imgs = rrc(imgs)
            imgs = hflip(imgs)
            imgs = color_jitter(imgs)

            lam = np.random.beta(1.0, 1.0)
            rand_idx = torch.randperm(imgs.size(0), device=device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_idx, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))

            outputs = model(imgs)
            loss = criterion(outputs, tgt) * ratio + criterion(outputs, tgt[rand_idx]) * (1. - ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s)")

    model.eval()
    correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(val_images), batch_size):
            imgs = val_images[start:start + batch_size]
            tgt = val_labels[start:start + batch_size]
            outputs = model(imgs)
            _, pred = outputs.max(1)
            total += tgt.size(0)
            correct += pred.eq(tgt).sum().item()
            if outputs.size(1) >= 5:
                _, pred5 = outputs.topk(5, 1, True, True)
                top5_correct += pred5.eq(tgt.view(-1, 1)).any(dim=1).sum().item()

    top1 = 100.0 * correct / total
    top5 = 100.0 * top5_correct / total if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")
    del model
    torch.cuda.empty_cache()
    return top1, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--class-file", type=str, default="./misc/class100.txt")
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.class_file) as f:
        all_classes = [l.strip() for l in f.readlines()]
    class_names = all_classes[:args.nclass]
    print(f"Classes ({args.nclass}): {class_names}")

    results = []
    for seed in args.seeds:
        print(f"\nSeed {seed}:")
        top1, top5 = evaluate_fast(
            args.train_dir, args.val_dir, class_names, args.nclass, device,
            img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size,
            seed=seed, depth=args.depth,
        )
        print(f"  Seed {seed}: Top-1={top1:.2f}%, Top-5={top5:.2f}%")
        results.append({"seed": seed, "top1": top1, "top5": top5})

    top1s = [r["top1"] for r in results]
    print(f"\n{'='*60}")
    print(f"Mean Top-1: {np.mean(top1s):.2f} ± {np.std(top1s):.2f}%")
    print(f"Mean Top-5: {np.mean([r['top5'] for r in results]):.2f} ± {np.std([r['top5'] for r in results]):.2f}%")


if __name__ == "__main__":
    main()
