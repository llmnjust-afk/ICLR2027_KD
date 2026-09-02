# AGS-DD Project Status
**Last Updated:** 2026-09-02 00:45 UTC

## Overview
AGS-DD (Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation) extends MGD3 (ICML 2025 Oral) with three adaptive modules: CAGS, IAST, and TAGS. Targeting ICLR 2027 (paper deadline Sep 25, 2026).

## Completed Experiments

### 10-class ImageNet-100 (ConvNet-6, IPC=10, 3 seeds, 1000 epochs)

#### Fixed Lambda
| Lambda | Top-1 (%) | Top-5 (%) |
|---|---|---|
| 0.0 (unguided) | 40.67 +/- 0.50 | 82.00 +/- 1.66 |
| 0.05 | **41.40 +/- 1.18** | 82.00 +/- 1.18 |
| 0.08 | 38.80 +/- 1.45 | 81.93 +/- 0.90 |
| 0.10 | 32.73 +/- 0.94 | 80.27 +/- 0.66 |
| 0.12 | 39.07 +/- 0.74 | 83.87 +/- 0.57 |
| 0.15 | 33.67 +/- 1.73 | 79.53 +/- 0.57 |

#### CAGS v1
| Range | Top-1 (%) | Top-5 (%) |
|---|---|---|
| [0.05, 0.12] | 38.27 +/- 1.64 | 81.40 +/- 0.43 |
| [0.05, 0.15] | 33.20 +/- 0.99 | 79.73 +/- 0.34 |
| [0.08, 0.12] | 37.07 +/- 1.23 | 79.47 +/- 0.41 |

#### CAGS v2
| Range | Top-1 (%) | Top-5 (%) |
|---|---|---|
| [0.0, 0.08] | 40.60 +/- 0.71 | 78.87 +/- 0.62 |
| [0.02, 0.06] | 39.80 +/- 0.59 | 77.73 +/- 0.34 |
| [0.0, 0.06] | 36.53 +/- 1.16 | 80.93 +/- 0.66 |
| [0.02, 0.08] | 33.67 +/- 1.75 | — |
| [0.0, 0.10] | 33.60 +/- 1.02 | — |
| inv [0.08, 0.0] | 37.60 +/- 2.14 | — |

#### TAGS (peak lambda=0.1)
| Schedule | Top-1 (%) | Top-5 (%) |
|---|---|---|
| cosine | 30.60 +/- 1.02 | 74.47 +/- 0.84 |
| linear | 35.53 +/- 0.25 | 78.00 +/- 0.59 |
| exponential | 35.27 +/- 1.37 | 79.67 +/- 0.19 |

### 100-class ImageNet-100 (ConvNet-6, IPC=10, 3 seeds, 1000 epochs)

| Config | Top-1 (%) | Top-5 (%) | vs Unguided |
|---|---|---|---|
| Unguided | 9.72 +/- 0.13 | 25.61 +/- 0.45 | — |
| Fixed lambda=0.10 | 10.07 +/- 0.15 | 27.50 +/- 0.48 | +0.35% |
| CAGS v1 [0.05, 0.12] | 10.79 +/- 0.62 | 28.54 +/- 0.81 | +1.07% |
| CAGS v1 [0.05, 0.15] | 10.38 +/- 0.17 | 28.51 +/- 0.44 | +0.66% |
| **CAGS v2 [0.0, 0.06]** | **12.57 +/- 0.22** | **30.29 +/- 0.56** | **+2.85% (29.3% rel.)** |
| CAGS v2 [0.02, 0.06] | 12.39 +/- 0.64 | 30.35 +/- 1.02 | +2.67% (27.5% rel.) |

### ImageNette (10 classes, ConvNet-6, IPC=10, 3 seeds, 1000 epochs)

| Config | Top-1 (%) | Top-5 (%) | vs Unguided |
|---|---|---|---|
| Unguided | 38.20 +/- 0.86 | 81.89 +/- 0.42 | — |
| Fixed lambda=0.05 | 36.98 +/- 0.37 | 80.11 +/- 0.52 | -1.22% |

### ImageWoof (10 classes, ConvNet-6, IPC=10, 3 seeds, 1000 epochs)

| Config | Top-1 (%) | Top-5 (%) | vs Unguided |
|---|---|---|---|
| Unguided | 20.23 +/- 0.29 | 65.85 +/- 0.78 | — |
| Fixed lambda=0.05 | 21.08 +/- 0.46 | 66.36 +/- 1.03 | +0.85% |

## Key Findings

1. CAGS v2 [0.0, 0.06] is the best config for 100-class: 12.57% vs 9.72% unguided (+29.3% relative)
2. CAGS benefit scales with class count: 10-class shows no benefit, 100-class shows huge benefit
3. Fixed guidance is dataset-dependent: helps on ImageWoof (+0.85%), hurts on ImageNette (-1.22%)
4. TAGS with lambda=0.1 hurts on 10-class: all schedules worse than unguided
5. Best fixed lambda on 10-class is 0.05: 41.40% vs 40.67% unguided (marginal +0.73%)
6. CAGS v2 outperforms v1 at 100 classes: 12.57% vs 10.79% (+16.5% relative)
7. Complexity direction matters: normal [0.0, 0.08] (40.60%) > inverted [0.08, 0.0] (37.60%) by 3.0%

## Remaining Experiments (Phase 3)

### High Priority
- [ ] IPC=1 experiments (10-class and 100-class)
- [ ] IPC=50 experiments (10-class and 100-class)
- [ ] ResNet-18 architecture (10-class and 100-class)
- [ ] ResNet-50 architecture (10-class and 100-class)
- [ ] 2000 epochs final runs (for reported numbers)

### Medium Priority
- [ ] FID computation for all generated datasets
- [ ] CAGS v2 on ImageNette/ImageWoof
- [ ] TAGS on 100-class (where CAGS works)
- [ ] IAST (IPC-Adaptive Stop Timing) experiments

## Paper Status

- Framework: Complete (main.tex + 6 section files + refs.bib + framework figure)
- Compiles: Yes (6.5MB PDF)
- References: 35 entries in refs.bib
- All sections have actual data filled in

## GitHub
- Repo: https://github.com/llmnjust-afk/ICLR2027_KD
- All experiment results pushed to GitHub
