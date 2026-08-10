"""Quick smoke test: generate 1 image with 5 steps to verify pipeline works."""
import os, sys, time
sys.argv = [
    "sample_ags.py", "--spec", "nette", "--num-samples", "1",
    "--nclass", "3", "--num-datasets", "1",
    "--imagenet-dir", "/ssd_data/imagenette/imagenette2/",
    "--save-dir", "./test_output/smoke",
    "--num-sampling-steps", "5", "--closest-point",
    "--no-cags", "--no-iast", "--no-tags",
]
exec(open("sample_ags.py").read())
