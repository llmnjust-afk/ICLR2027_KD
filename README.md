# AGS-DD: Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation

This repository implements **AGS-DD**, a novel framework for training-free diffusion-based dataset distillation that dynamically adjusts guidance strategies based on class complexity, IPC settings, and diffusion timesteps.

Built on top of [MGD³](https://github.com/jachansantiago/mode_guidance) (ICML 2025 Oral).

## Overview

Existing training-free diffusion dataset distillation methods (MGD³, CoDA, DMGD) use fixed guidance parameters across all classes, IPC settings, and timesteps. AGS-DD addresses this limitation with three adaptive modules:

### Three Core Modules

| Module | Full Name | Function |
|--------|-----------|----------|
| **CAGS** | Class-Adaptive Guidance Strength | Adjusts guidance strength per-class based on intra-class complexity (mode count + entropy) |
| **IAST** | IPC-Adaptive Stop Timing | Determines when to stop mode guidance based on IPC and class complexity |
| **TAGS** | Timestep-Adaptive Guidance Scheduling | Varies guidance weight across diffusion timesteps (strong at high-noise, weak at low-noise) |

### Key Properties

- **Training-free**: Uses pre-trained diffusion models without fine-tuning
- **Adaptive**: Parameters adjust per-class, per-IPC, per-timestep
- **Universal**: Works with DiT, LDM, and Stable Diffusion models
- **Efficient**: Minimal computational overhead vs. baseline MGD³

## Repository Structure

```
ICLR2027_KD/
├── ags/                        # AGS-DD core modules
│   ├── __init__.py
│   ├── class_complexity.py     # CAGS: Class-Adaptive Guidance Strength
│   ├── adaptive_stop.py        # IAST: IPC-Adaptive Stop Timing
│   ├── guidance_schedule.py    # TAGS: Timestep-Adaptive Guidance Scheduling
│   └── ags_sampler.py          # AGS Sampler: Integrates all three modules
├── evaluation/                 # Evaluation code
│   ├── __init__.py
│   ├── train_eval.py           # Train & evaluate on distilled datasets
│   ├── cross_arch.py           # Cross-architecture generalization evaluation
│   └── metrics.py              # Metrics computation and formatting
├── configs/                    # Configuration files
│   └── default.yaml            # Default AGS-DD configuration
├── scripts/                    # Run scripts
│   ├── run_imagenette.sh       # Generate ImageNette dataset
│   ├── run_imagewoof.sh        # Generate ImageWoof dataset
│   ├── run_imagenet1k.sh       # Generate ImageNet-1K dataset
│   ├── run_cross_arch.sh       # Cross-architecture evaluation
│   └── run_ablation.sh         # Ablation study
├── sample_ags.py               # Main entry point for AGS-DD sampling
├── run_ablation.py             # Ablation study runner
├── README.md                   # This file
└── requirements.txt            # Python dependencies
│
│   ===== MGD³ Baseline Files (inherited) =====
├── diffusion/                  # Diffusion model implementation
├── diffusers/                  # Diffusers library (modified)
├── models.py                   # DiT model definitions
├── data.py                     # Data loading
├── argument.py                 # Argument parser (baseline)
├── sample.py                   # Baseline sampling (no guidance)
├── sample_mode_guidance.py     # MGD³ mode guidance sampling
├── train.py                    # Model training
├── train_dit.py                # DiT training
├── finetune_dit.py             # DiT fine-tuning
├── tsne_plots.py               # Feature extraction & visualization
├── download.py                 # Model download utilities
└── misc/                       # Class lists and utilities
```

## Installation

```bash
# Clone this repository
git clone https://github.com/llmnjust-afk/ICLR2027_KD.git
cd ICLR2027_KD

# Create conda environment
conda create -n ags_dd python=3.8
conda activate ags_dd

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate Distilled Dataset

```bash
# ImageNette (10 classes, fast for debugging)
bash scripts/run_imagenette.sh 10  # IPC=10

# ImageWoof (10 classes, fine-grained)
bash scripts/run_imagewoof.sh 50  # IPC=50

# ImageNet-1K (1000 classes, run in phases)
bash scripts/run_imagenet1k.sh 50 0  # IPC=50, Phase 0 (classes 0-99)
```

### 2. Evaluate on Real Test Set

```bash
# Single architecture evaluation
python -m evaluation.train_eval \
    --train-dir ./generated/ags_dd/nette_ipc10/dataset_0 \
    --test-dir /path/to/imagenet/val \
    --arch convnet7 \
    --num-classes 10 \
    --epochs 2000 \
    --num-seeds 5 \
    --save-dir ./results \
    --dataset-name imagenette \
    --ipc 10
```

### 3. Cross-Architecture Evaluation

```bash
bash scripts/run_cross_arch.sh \
    ./generated/ags_dd/nette_ipc10/dataset_0 \
    /path/to/imagenet/val \
    10
```

### 4. Ablation Study

```bash
bash scripts/run_ablation.sh /path/to/imagenet/val
```

### 5. Direct Python Usage

```python
from ags import ClassComplexityAnalyzer, AdaptiveStopTiming, TimestepAdaptiveSchedule, AGSSampler

# Initialize modules
complexity = ClassComplexityAnalyzer(n_clusters_range=(2, 20), alpha=0.5, beta=0.5)
stop_timing = AdaptiveStopTiming(t_max=50, lam=0.1)
schedule = TimestepAdaptiveSchedule(w_max=0.5, schedule_type="cosine")

# Create sampler
sampler = AGSSampler(
    model=model, vae=vae, diffusion=diffusion,
    complexity_analyzer=complexity,
    adaptive_stop=stop_timing,
    guidance_schedule=schedule,
    device="cuda",
)

# Generate dataset
sampler.generate_dataset(args, class_labels, sel_classes, clusters_centers, save_dir)
```

## Ablation Configurations

The ablation study tests 8 configurations:

| Config | CAGS | IAST | TAGS |
|--------|------|------|------|
| baseline | ✗ | ✗ | ✗ |
| cags_only | ✓ | ✗ | ✗ |
| iast_only | ✗ | ✓ | ✗ |
| tags_only | ✗ | ✗ | ✓ |
| cags_iast | ✓ | ✓ | ✗ |
| cags_tags | ✓ | ✗ | ✓ |
| iast_tags | ✗ | ✓ | ✓ |
| **full_ags_dd** | **✓** | **✓** | **✓** |

## Supported Architectures

| Architecture | Key | Description |
|-------------|-----|-------------|
| ConvNet-7 | `convnet7` | Standard DD evaluation network (128 channels) |
| ResNet-18 | `resnet18` | Standard CNN |
| ViT-Tiny | `vit_tiny` | Vision Transformer |
| Swin-Tiny | `swin_tiny` | Swin Transformer |
| DeiT-Tiny | `deit_tiny` | Data-efficient Image Transformer |

## Datasets

| Dataset | Resolution | Classes | Description |
|---------|-----------|---------|-------------|
| ImageNette | 224×224 | 10 | Easy ImageNet subset |
| ImageWoof | 224×224 | 10 | Fine-grained (dog breeds) |
| ImageNet-100 | 224×224 | 100 | Medium-scale subset |
| ImageNet-1K | 224×224 | 1000 | Full ImageNet |

## Citation

```bibtex
@inproceedings{agsdd2027,
  title={Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2027}
}
```

## Acknowledgments

This code is built upon [MGD³](https://github.com/jachansantiago/mode_guidance) (ICML 2025 Oral).
We thank the authors for their open-source contribution.
