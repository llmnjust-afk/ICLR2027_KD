#!/usr/bin/env python3
"""Create Random Herding baseline: randomly select IPC real images per class from ImageNet-100 train."""
import os
import shutil
import random
import argparse

def create_random_herding(src_dir, dst_dir, ipc, class_file, nclass, seed=42):
    random.seed(seed)
    with open(class_file) as f:
        classes = [l.strip() for l in f.readlines()][:nclass]
    
    os.makedirs(os.path.join(dst_dir, "dataset_0"), exist_ok=True)
    
    for cls in classes:
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, "dataset_0", cls)
        os.makedirs(dst_cls, exist_ok=True)
        
        if not os.path.isdir(src_cls):
            print(f"WARNING: {src_cls} not found, skipping")
            continue
        
        all_imgs = sorted(os.listdir(src_cls))
        selected = random.sample(all_imgs, min(ipc, len(all_imgs)))
        
        for img in selected:
            shutil.copy2(os.path.join(src_cls, img), os.path.join(dst_cls, img))
    
    total = sum(len(os.listdir(os.path.join(dst_dir, "dataset_0", c))) for c in classes if os.path.isdir(os.path.join(dst_dir, "dataset_0", c)))
    print(f"Created Random Herding dataset: {total} images ({ipc} per class × {nclass} classes)")
    print(f"  Source: {src_dir}")
    print(f"  Dest: {dst_dir}/dataset_0/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, required=True, help="Source training directory (e.g., /root/data/imagenet100/train)")
    parser.add_argument("--dst-dir", type=str, required=True, help="Destination directory for herding dataset")
    parser.add_argument("--class-file", type=str, required=True)
    parser.add_argument("--nclass", type=int, default=100)
    parser.add_argument("--ipc", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    create_random_herding(args.src_dir, args.dst_dir, args.ipc, args.class_file, args.nclass, args.seed)
