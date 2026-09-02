# AGS-DD Project Status

## Last Updated: Sep 2, 2026 22:45 UTC

## Current Phase: Best-Accuracy Reruns (IN PROGRESS)

### Key Changes from Previous Version
1. **Fixed training hyperparameters** to match MGD³: lr=0.1, wd=1e-4, bs=128
2. **Best-accuracy tracking** (matching MGD³'s eval protocol — reports best acc across all epochs, not final)
3. **Correct epoch counts** per MGD³'s ipc_epoch() formula:
   - IPC=1, 10-class: 3000 epochs
   - IPC=1, 100-class: 2000 epochs
   - IPC=10, 10-class: 2000 epochs
   - IPC=10, 100-class: 1300 epochs
   - IPC=50, 10-class: 1500 epochs
   - IPC=50, 100-class: 1000 epochs
4. **Random Herding baseline** implemented and datasets created (eval pending)

### Results So Far (best-acc, lr=0.1)

#### GPU1 (10-class + ImageNette/Woof) — 3/12 configs done
| Config | Top-1 (%) |
|--------|-----------|
| 10-class unguided | 39.07 ± 1.16% |
| 10-class fixed λ=0.05 | 41.87 ± 0.25% |
| 10-class CAGS v2 [0.0,0.08] | 40.87 ± 0.98% |
| 10-class IPC=50 unguided | IN PROGRESS |

#### GPU0 (100-class + ResNet-18) — seed 0 of config 1 done
| Config | Top-1 (%) |
|--------|-----------|
| 100-class unguided (seed 0) | 12.06% |

### Pending Experiments
- GPU0: 100-class unguided/fixed/CAGS/TAGS+CAGS/IAST+CAGS (IPC=10), IPC=50, IPC=1, ResNet-18
- GPU1: 10-class IPC=50/1, ImageNette, ImageWoof
- Random Herding baselines (all datasets, all IPCs)
- v3 normalization test (correct augmentation order + Lighting)

### MGD³ Published Numbers (for comparison)
- ImageNet-100 ConvNet-6 IPC=10: Random=17.0%, MinMax=24.3%, MGD³=25.8%
- ImageNette ConvNet-6 IPC=10: DiT-only=53.2%, MGD³=59.6%
- ImageNette ResNetAP-10 IPC=10: DiT-only=57.1%, MGD³=66.4%

### Known Differences from MGD³ Protocol
1. Sampling steps: 25 (ours) vs 50 (MGD³)
2. Missing Lighting augmentation (v3 script prepared with fix)
3. Normalization order: was before augmentation (fixed in v3)
4. Image resolution: 224×224 (ours) vs 256×256 (MGD³)
