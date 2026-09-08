#!/usr/bin/env python3
"""Class-level mechanism analysis: correlate complexity scores with per-class accuracy gains.

Trains unguided and CAGS models (seed 0), extracts per-class accuracy,
then computes Pearson/Spearman correlations between:
  - complexity score → accuracy gain (CAGS - unguided)
  - lambda → accuracy gain
  - individual factors → accuracy gain

Outputs:
  - class_level_analysis.json
  - figures/class_complexity_analysis.pdf (3-panel scatter + histogram)
"""
import os, sys, json, glob, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_eval_v3 import (
    create_model, load_images_to_tensor, load_val_data, rand_bbox, Lighting
)

def train_and_eval_per_class(train_dir, val_dir, class_names, num_classes, device,
                              img_size=224, epochs=1300, batch_size=128, seed=0,
                              depth=6, arch="convnet", norm_type="instance",
                              lr=0.1, weight_decay=1e-4):
    """Train model and return per-class accuracy."""
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

    model = create_model(arch, num_classes, depth=depth, norm_type=norm_type, img_size=img_size)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_images)
    rrc = transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True)
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(alphastd=0.1)

    best_top1 = 0.0
    best_state = None
    eval_interval = max(epochs // 100, 1)

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
            imgs = lighting(imgs)
            imgs = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for start in range(0, len(val_images), batch_size):
                    imgs = val_images[start:start + batch_size]
                    tgt = val_labels[start:start + batch_size]
                    outputs = model(imgs)
                    _, pred = outputs.max(1)
                    total += tgt.size(0)
                    correct += pred.eq(tgt).sum().item()
            cur_top1 = 100.0 * correct / total
            if cur_top1 > best_top1:
                best_top1 = cur_top1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{epochs} ({time.time()-t0:.1f}s) best={best_top1:.2f}%")

    # Load best model and compute per-class accuracy
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    with torch.no_grad():
        for start in range(0, len(val_images), batch_size):
            imgs = val_images[start:start + batch_size]
            tgt = val_labels[start:start + batch_size]
            outputs = model(imgs)
            _, pred = outputs.max(1)
            for i in range(tgt.size(0)):
                label = tgt[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1
    
    per_class_acc = {}
    for c in range(num_classes):
        per_class_acc[c] = class_correct[c] / max(class_total[c], 1)
    
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s, best Top-1={best_top1:.2f}%")
    del model
    torch.cuda.empty_cache()
    return per_class_acc, best_top1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="./class_level_analysis.json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load complexity stats
    with open("./results/sweep_in100/complexity_stats.json") as f:
        stats = json.load(f)
    
    complexity_scores = {int(k): v for k, v in stats["complexity_scores"].items()}
    mode_counts = {int(k): v for k, v in stats["mode_counts"].items()}
    intra_variances = {int(k): v for k, v in stats["intra_variances"].items()}
    separabilities = {int(k): v for k, v in stats["separabilities"].items()}
    
    # Compute per-class lambda for CAGS [0.0, 0.06]
    lambda_min, lambda_max = 0.0, 0.06
    per_class_lambda = {}
    for c, score in complexity_scores.items():
        per_class_lambda[c] = lambda_min + (lambda_max - lambda_min) * score
    
    # Load class names
    with open("./misc/class100.txt") as f:
        all_classes = [l.strip() for l in f.readlines()]
    class_names = all_classes[:100]
    
    # Train unguided
    print("\n=== Training Unguided model (seed {}) ===".format(args.seed))
    unguided_data = "./results/sweep_in100/high_noise_unguided_d25/dataset_0"
    unguided_acc, unguided_top1 = train_and_eval_per_class(
        unguided_data, "/root/data/imagenet100/val", class_names, 100, device,
        epochs=1300, seed=args.seed)
    
    # Train CAGS
    print("\n=== Training CAGS model (seed {}) ===".format(args.seed))
    cags_data = "./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0"
    cags_acc, cags_top1 = train_and_eval_per_class(
        cags_data, "/root/data/imagenet100/val", class_names, 100, device,
        epochs=1300, seed=args.seed)
    
    # Compute correlations
    from scipy.stats import pearsonr, spearmanr
    
    classes = sorted(complexity_scores.keys())
    complexities = np.array([complexity_scores[c] for c in classes])
    lambdas = np.array([per_class_lambda[c] for c in classes])
    unguided_accs = np.array([unguided_acc[c] for c in classes])
    cags_accs = np.array([cags_acc[c] for c in classes])
    gains = cags_accs - unguided_accs
    mode_counts_arr = np.array([mode_counts[c] for c in classes])
    intra_vars = np.array([intra_variances[c] for c in classes])
    seps = np.array([separabilities[c] for c in classes])
    
    results = {}
    results["unguided_top1"] = float(unguided_top1)
    results["cags_top1"] = float(cags_top1)
    
    # Complexity vs gain
    r, p = pearsonr(complexities, gains)
    rho, p_rho = spearmanr(complexities, gains)
    results["complexity_vs_gain"] = {"pearson_r": float(r), "pearson_p": float(p),
                                      "spearman_rho": float(rho), "spearman_p": float(p_rho)}
    print(f"\nComplexity vs Accuracy Gain: r={r:.4f} (p={p:.4f}), rho={rho:.4f} (p={p_rho:.4f})")
    
    # Lambda vs gain
    r2, p2 = pearsonr(lambdas, gains)
    results["lambda_vs_gain"] = {"pearson_r": float(r2), "pearson_p": float(p2)}
    print(f"Lambda vs Accuracy Gain: r={r2:.4f} (p={p2:.4f})")
    
    # Individual factors
    for name, arr in [("mode_count", mode_counts_arr), ("intra_var", intra_vars), ("separability", seps)]:
        r_f, p_f = pearsonr(arr, gains)
        results[f"{name}_vs_gain"] = {"pearson_r": float(r_f), "pearson_p": float(p_f)}
        print(f"{name} vs Gain: r={r_f:.4f} (p={p_f:.4f})")
    
    # Top/bottom 5 by complexity
    sorted_idx = np.argsort(complexities)
    results["most_complex_5"] = [int(classes[i]) for i in sorted_idx[-5:]][::-1]
    results["least_complex_5"] = [int(classes[i]) for i in sorted_idx[:5]]
    
    print(f"\nMost complex 5: {results['most_complex_5']}")
    print(f"  Complexities: {[f'{complexities[i]:.4f}' for i in sorted_idx[-5:][::-1]]}")
    print(f"  Gains: {[f'{gains[i]:.4f}' for i in sorted_idx[-5:][::-1]]}")
    print(f"\nLeast complex 5: {results['least_complex_5']}")
    print(f"  Complexities: {[f'{complexities[i]:.4f}' for i in sorted_idx[:5]]}")
    print(f"  Gains: {[f'{gains[i]:.4f}' for i in sorted_idx[:5]]}")
    
    # Save raw data
    results["raw_data"] = {
        "classes": [int(c) for c in classes],
        "complexity_scores": [float(x) for x in complexities],
        "lambda_values": [float(x) for x in lambdas],
        "unguided_acc": [float(x) for x in unguided_accs],
        "cags_acc": [float(x) for x in cags_accs],
        "accuracy_gain": [float(x) for x in gains],
        "mode_counts": [int(x) for x in mode_counts_arr],
        "intra_variances": [float(x) for x in intra_vars],
        "separabilities": [float(x) for x in seps],
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")
    
    # Generate scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    ax = axes[0]
    ax.scatter(complexities, gains, alpha=0.6, s=30, c='steelblue')
    z = np.polyfit(complexities, gains, 1)
    ax.plot(complexities, np.polyval(z, complexities), "r--", alpha=0.8)
    ax.set_xlabel("Class Complexity Score")
    ax.set_ylabel("Per-class Accuracy Gain (CAGS $-$ Unguided)")
    ax.set_title(f"Complexity vs Gain ($r$={r:.3f}, $p$={p:.3f})")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.2)
    
    ax = axes[1]
    ax.scatter(lambdas, gains, alpha=0.6, s=30, c='darkorange')
    z2 = np.polyfit(lambdas, gains, 1)
    ax.plot(lambdas, np.polyval(z2, lambdas), "r--", alpha=0.8)
    ax.set_xlabel("Per-class Guidance Strength ($\\lambda_c$)")
    ax.set_ylabel("Per-class Accuracy Gain")
    ax.set_title(f"$\\lambda_c$ vs Gain ($r$={r2:.3f}, $p$={p2:.3f})")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.2)
    
    ax = axes[2]
    ax.hist(gains, bins=20, color='seagreen', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
    ax.axvline(x=np.mean(gains), color='blue', linestyle='-', alpha=0.7, label=f'Mean={np.mean(gains):.3f}')
    ax.set_xlabel("Per-class Accuracy Gain")
    ax.set_ylabel("Number of Classes")
    ax.set_title("Distribution of Per-class Gains")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    os.makedirs("./figures", exist_ok=True)
    plot_path = "./figures/class_complexity_analysis.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
