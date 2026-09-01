#!/usr/bin/env python3
"""Patch duration_sweep.py to add alpha/beta/gamma/delta CLI args."""

with open('/root/ICLR2027_KD/duration_sweep.py', 'r') as f:
    content = f.read()

# --- Add alpha/beta/gamma/delta to compute_clusters signature ---
old_sig = "def compute_clusters(spec, imagenet_dir, vae, device, nclass=10,\n                        sigmoid_slope=3.0, sigmoid_center=0.6, complexity_k=None):"
new_sig = "def compute_clusters(spec, imagenet_dir, vae, device, nclass=10,\n                        sigmoid_slope=3.0, sigmoid_center=0.6, complexity_k=None,\n                        alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):"
assert old_sig in content, "compute_clusters signature not found"
content = content.replace(old_sig, new_sig)

# --- Pass alpha/beta/gamma/delta to ClassComplexityAnalyzer ---
old_init = """    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=0.3, beta=0.3, gamma=0.2, delta=0.2,
        use_pca=True, sigmoid_slope=sigmoid_slope, sigmoid_center=sigmoid_center,
    )"""
new_init = """    analyzer = ClassComplexityAnalyzer(
        n_clusters_range=(2, 20), alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        use_pca=True, sigmoid_slope=sigmoid_slope, sigmoid_center=sigmoid_center,
    )"""
assert old_init in content, "ClassComplexityAnalyzer init not found"
content = content.replace(old_init, new_init)

# --- Add CLI args ---
old_args = """    parser.add_argument("--complexity-k", type=int, default=None,
                        help="Fixed K for complexity computation (default: silhouette selection)")"""
new_args = """    parser.add_argument("--complexity-k", type=int, default=None,
                        help="Fixed K for complexity computation (default: silhouette selection)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight for mode_count in complexity (default 0.3)")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight for entropy in complexity (default 0.3)")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="Weight for intra-class variance in complexity (default 0.2)")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="Weight for 1-separability in complexity (default 0.2)")"""
assert old_args in content, "complexity-k arg not found"
content = content.replace(old_args, new_args)

# --- Pass alpha/beta/gamma/delta to compute_clusters call ---
old_call = """            analyzer = compute_clusters(args.spec, args.imagenet_dir, vae, device,
                                        nclass=args.nclass,
                                        sigmoid_slope=args.sigmoid_slope,
                                        sigmoid_center=args.sigmoid_center,
                                        complexity_k=args.complexity_k)"""
new_call = """            analyzer = compute_clusters(args.spec, args.imagenet_dir, vae, device,
                                        nclass=args.nclass,
                                        sigmoid_slope=args.sigmoid_slope,
                                        sigmoid_center=args.sigmoid_center,
                                        complexity_k=args.complexity_k,
                                        alpha=args.alpha, beta=args.beta,
                                        gamma=args.gamma, delta=args.delta)"""
assert old_call in content, "compute_clusters call not found"
content = content.replace(old_call, new_call)

with open('/root/ICLR2027_KD/duration_sweep.py', 'w') as f:
    f.write(content)

print("Patched duration_sweep.py with alpha/beta/gamma/delta CLI args!")
