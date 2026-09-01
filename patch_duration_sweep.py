#!/usr/bin/env python3
"""Patch duration_sweep.py to pass sigmoid params and complexity_k to compute_clusters."""

import re

with open('/root/ICLR2027_KD/duration_sweep.py', 'r') as f:
    content = f.read()

# --- Fix 1: compute_clusters signature ---
old_sig = "def compute_clusters(spec, imagenet_dir, vae, device, nclass=10):"
new_sig = "def compute_clusters(spec, imagenet_dir, vae, device, nclass=10,\n                        sigmoid_slope=3.0, sigmoid_center=0.6, complexity_k=None):"
assert old_sig in content, "compute_clusters signature not found"
content = content.replace(old_sig, new_sig)

# --- Fix 2: Pass sigmoid params to ClassComplexityAnalyzer ---
old_init = """    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=0.3, beta=0.3, gamma=0.2, delta=0.2,
        use_pca=True, sigmoid_slope=3.0, sigmoid_center=0.6,
    )
    analyzer.analyze_all_classes(features_per_class, paths_per_class)"""
new_init = """    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=0.3, beta=0.3, gamma=0.2, delta=0.2,
        use_pca=True, sigmoid_slope=sigmoid_slope, sigmoid_center=sigmoid_center,
    )
    analyzer.analyze_all_classes(features_per_class, paths_per_class, fixed_k=complexity_k)"""
assert old_init in content, "ClassComplexityAnalyzer init not found"
content = content.replace(old_init, new_init)

# --- Fix 3: Add --regen-cache and --complexity-k CLI args ---
old_args = """    parser.add_argument("--sigmoid-center", type=float, default=0.6,
                        help="Sigmoid center for CAGS complexity mapping (default 0.6)")
    args = parser.parse_args()"""
new_args = """    parser.add_argument("--sigmoid-center", type=float, default=0.6,
                        help="Sigmoid center for CAGS complexity mapping (default 0.6)")
    parser.add_argument("--regen-cache", action="store_true", default=False,
                        help="Delete existing cluster_cache.pkl and regenerate")
    parser.add_argument("--complexity-k", type=int, default=None,
                        help="Fixed K for complexity computation (default: silhouette selection)")
    args = parser.parse_args()"""
assert old_args in content, "sigmoid-center arg not found"
content = content.replace(old_args, new_args)

# --- Fix 4: Delete cache if --regen-cache, pass params to compute_clusters ---
old_cache = """        if os.path.isfile(cluster_cache):
            print(f"\\nLoading cached CAGS clusters from {cluster_cache}")
            with open(cluster_cache, "rb") as f:
                analyzer = pickle.load(f)
        else:
            print("\\nComputing CAGS clusters...")
            analyzer = compute_clusters(args.spec, args.imagenet_dir, vae, device, nclass=args.nclass)
            with open(cluster_cache, "wb") as f:
                pickle.dump(analyzer, f)
            print(f"Cluster cache saved to {cluster_cache}")"""
new_cache = """        if args.regen_cache and os.path.isfile(cluster_cache):
            print(f"\\n[regen-cache] Deleting {cluster_cache}")
            os.remove(cluster_cache)

        if os.path.isfile(cluster_cache) and not args.regen_cache:
            print(f"\\nLoading cached CAGS clusters from {cluster_cache}")
            with open(cluster_cache, "rb") as f:
                analyzer = pickle.load(f)
        else:
            print("\\nComputing CAGS clusters...")
            analyzer = compute_clusters(args.spec, args.imagenet_dir, vae, device,
                                        nclass=args.nclass,
                                        sigmoid_slope=args.sigmoid_slope,
                                        sigmoid_center=args.sigmoid_center,
                                        complexity_k=args.complexity_k)
            with open(cluster_cache, "wb") as f:
                pickle.dump(analyzer, f)
            print(f"Cluster cache saved to {cluster_cache}")"""
assert old_cache in content, "cache loading block not found"
content = content.replace(old_cache, new_cache)

with open('/root/ICLR2027_KD/duration_sweep.py', 'w') as f:
    f.write(content)

print("Patched duration_sweep.py successfully!")
print("Changes:")
print("  1. compute_clusters now accepts sigmoid_slope, sigmoid_center, complexity_k")
print("  2. Added --regen-cache CLI flag")
print("  3. Added --complexity-k CLI flag")
print("  4. Cache regenerated when --regen-cache is set")
