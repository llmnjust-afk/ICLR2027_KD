# AGS-DD 项目状态总结

> **项目**: Adaptive Guidance Scheduling for Training-Free Diffusion Dataset Distillation  
> **目标会议**: ICLR 2027  
> **基础方法**: MGD³ (Mode-Guided Dataset Distillation, ICML 2025 Oral)  
> **GitHub**: `https://github.com/llmnjust-afk/ICLR2027_KD`  
> **最后更新**: 2026-09-01

---

## 1. 项目概述

AGS-DD 在 MGD³ 的基础上提出三个自适应模块，替代 MGD³ 中固定的超参数：

| 模块 | 全称 | 替代的 MGD³ 固定参数 | 核心思想 |
|:-:|:-:|:-:|:-:|
| **CAGS** | Class-Adaptive Guidance Strength | 固定 `mode_guidance_scale=0.1` | 复杂类需更强引导，简单类弱引导 |
| **IAST** | IPC-Adaptive Stop Timing | 固定 `stop_t=25` | IPC 越大、模式越多，引导持续越久 |
| **TAGS** | Timestep-Adaptive Guidance Schedule | 恒定引导权重 | 引导强度随 timestep 余弦变化 |

**引导公式**（与 MGD³ 完全一致，在 `ags/ags_sampler.py:231-239`）：

```python
guidance_score = -(xstart_cond - mode_features) * w_t * torch.exp(0.5 * out["log_variance"])
img = out["mean"] + guidance_score + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
```

其中 `w_t` 由 TAGS 调度，`mode_features` 由 CAGS 聚类中心决定，引导停止时机由 IAST 决定。

---

## 2. 关键 Bug 修复记录

### Bug 1: 评估 epoch 不足
- **问题**: 初始用 20 epochs 评估，结果不稳定且偏低
- **修复**: 改为 1000 epochs（MGD³ 标准为 2000）

### Bug 2: Mode features 计算方式不匹配
- **问题**: 使用 PCA + closest_point 方式计算 mode features，与 MGD³ 不一致
- **修复**: `compute_clusters_for_ipc(ipc, use_pca=False, closest_point=False)` — 直接用 K-means 质心

### Bug 3: 引导窗口方向错误
- **问题**: low_noise/high_noise 窗口方向搞反
- **修复**: 确认 high_noise = 引导在去噪过程前段（高噪声区，t→t_max）

### Bug 4（最关键）: Cluster Center 映射错位
- **根因**: `ags/class_complexity.py` 的 `compute_clusters_for_ipc` 使用 `seq_idx`（enumerate 序号）而非 `class_label` 作为字典键。由于 `features_per_class` 的键顺序为 `[1,9,5,3,2,0,8,6,7,4]`（由数据加载顺序决定），导致每个类别取到了**错误类别的聚类中心**，引导将图像推向错误的 mode
- **修复**: `for class_label, features in self.features_per_class.items()` 直接用 `class_label` 作为键（匹配 MGD³ 官方 `clusters_centers[c] = kmeans.cluster_centers_`）
- **验证**: 修复前 30.00%（引导有害），修复后 39.00%（与 MGD³ 官方 39.70% 一致）

### 修复文件
- `ags/class_complexity.py` — `compute_clusters_for_ipc` 函数（第 269-293 行）
- `duration_sweep.py` — 评估参数对齐 MGD³（ConvNet depth=6, CutMix, RandomResizedCrop, MultiStepLR, batch_size=64, persistent_workers=True）

---

## 3. 实验结果汇总

### 3.1 10 类 ImageNet-100（Bug 修复验证, IPC=10, 1000 epochs, ConvNet-6, CutMix）

| 配置 | Top-1 | Top-5 | 说明 |
|:-:|:-:|:-:|:-:|
| MGD³ unguided（官方代码） | 37.70% | 77.70% | 基准 |
| MGD³ guided（官方代码） | 39.70% | 83.70% | 引导有效 (+2.0%) |
| 我们的 guided（修复前） | 30.00% | 76.60% | 引导有害 ❌ |
| **我们的 guided（修复后）** | **39.00 ± 0.40%** | **82.20%** | 引导恢复有效 ✅ |

**结论**: Bug 修复后，我们的实现与 MGD³ 官方效果一致，验证了代码正确性。

### 3.2 100 类 ImageNet-100（修复后, IPC=10, 1000 epochs, ConvNet-6, CutMix, 2 seeds）

| 配置 | Top-1 | Top-5 | vs Unguided |
|:-:|:-:|:-:|:-:|
| Unguided baseline | 9.70 ± 0.34% | 26.23 ± 0.13% | — |
| CAGS adaptive（λ≈0.14-0.19） | 9.91 ± 0.03% | 27.54 ± 0.16% | +0.21% |
| **MGD³ fixed（λ=0.1）** | **10.64 ± 0.10%** | **27.26 ± 0.18%** | **+0.94%** |

**结论**: 引导有效（+0.94%），但 CAGS 自适应强度过高（0.14-0.19 vs 固定 0.1），反而不如固定 0.1。

### 3.3 MGD³ 官方实验设置确认

从 MGD³ 的 `imagenet100.sh` 脚本发现：
- **`--nclass 10`** — 只使用 class100.txt 的前 10 个类别（不是全部 100 类）
- **2000 epochs**（`ipc_epoch(10, 1, 10)` = 2000）
- **ConvNet depth=6**, batch_size=64
- **CutMix**（不是 Mixup）, **RandomResizedCrop**（scale=(0.5, 1.0)）
- **MultiStepLR**（milestones=[2/3, 5/6], gamma=0.2）
- 不传 `--use_pca` 和 `--closest_point`

---

## 4. 代码架构

### 4.1 核心文件

```
ICLR2027_KD/
├── ags/                        # AGS-DD 三个自适应模块
│   ├── __init__.py
│   ├── class_complexity.py     # CAGS: 类复杂度分析 → 引导强度
│   ├── adaptive_stop.py        # IAST: IPC 自适应停止时机
│   ├── guidance_schedule.py    # TAGS: timestep 自适应调度
│   └── ags_sampler.py         # 统一采样器（集成三模块）
│
├── duration_sweep.py          # 主实验脚本（生成 + 评估）
├── quick_eval.py               # 快速评估脚本（GPU 预加载，~95s/seed/1000ep）
├── sample_mode_guidance.py     # MGD³ 原始采样代码（对照用）
├── train.py                    # MGD³ 原始训练评估代码
│
├── diffusion/                  # 扩散过程（与 MGD³ 完全相同）
│   ├── gaussian_diffusion.py
│   ├── diffusion_utils.py
│   └── __init__.py
│
├── models.py                   # DiT 模型定义（与 MGD³ 相同）
├── data.py                     # 数据加载（与 MGD³ 相同）
├── tsne_plots.py               # 特征提取（与 MGD³ 相同）
├── download.py                 # 模型下载工具
│
├── evaluation/
│   ├── train_eval.py           # ConvNet 训练评估（旧版）
│   ├── cross_arch.py           # 跨架构评估（未运行）
│   └── metrics.py              # FID 等指标（未实现）
│
├── misc/
│   ├── class100.txt            # ImageNet-100 类别列表
│   ├── class_indices.txt       # ImageNet-1000 全类别
│   ├── class_nette.txt         # ImageNette 类别
│   └── class_woof.txt          # ImageWoof 类别
│
├── scripts/
│   └── imagenet100.sh          # MGD³ 官方实验脚本
│
├── pretrained_models/          # DiT checkpoint（不入 git）
│   └── DiT-XL-2-256x256.pt
│
└── results/                    # 实验结果（不入 git）
    ├── sweep_in10/             # 10 类实验
    │   └── cluster_cache.pkl   # 聚类缓存
    └── sweep_in100/            # 100 类实验
        └── cluster_cache.pkl
```

### 4.2 关键参数

#### 生成参数（`duration_sweep.py`）

| 参数 | 默认值 | 说明 |
|:-:|:-:|:-:|
| `--spec imagenet100` | nette | 数据集规格 |
| `--nclass 10` | 10 | 类别数（MGD³ 用 10，非 100） |
| `--ipc 10` | 10 | 每类图像数 |
| `--window high_noise` | low_noise | 引导窗口（high_noise = 前段引导） |
| `--durations 25` | [10,15,25,35] | 引导持续步数 |
| `--schedule constant` | constant | 权重调度类型 |
| `--no-cags` | False | 禁用 CAGS，用固定 λ |
| `--fixed-scale 0.1` | 0.1 | 固定引导强度（MGD³ 默认） |
| `--fixed-stop-t 25` | None | 固定停止时间（MGD³ 默认） |
| `--depth 6` | 6 | ConvNet 深度 |
| `--epochs 1000` | 20 | 评估训练 epochs |

#### CAGS 参数（`ags/class_complexity.py`）

| 参数 | 值 | 说明 |
|:-:|:-:|:-:|
| `n_clusters_range` | (2, 20) | K-means 聚类数范围 |
| `alpha` | 0.5 | mode count 权重 |
| `beta` | 0.5 | entropy 权重 |
| `guidance_scale_range` | (0.05, 0.3) | CAGS 输出范围 |
| sigmoid 中心 | 0.5 | 复杂度归一化中心 |

复杂度公式: `Complexity(c) = sigmoid(5 * (α·K/max_K + β·entropy - 0.5))`  
引导强度: `strength = 0.05 + complexity * (0.3 - 0.05)` → 范围 [0.05, 0.30]

#### IAST 参数（`ags/adaptive_stop.py`）

| 参数 | 值 | 说明 |
|:-:|:-:|:-:|
| `t_max` | 50 | 最大 timestep |
| `lam` | 0.316 | 指数衰减率 |
| `min_stop` | 5 | 最小引导步数 |
| `max_stop_ratio` | 0.9 | 最大引导比例 |
| `complexity_weight` | 0.3 | 复杂度权重 |

停止时机公式: `guidance_steps = t_max * (1 - exp(-λ·√IPC / K(c))) + 0.3·complexity·t_max`

#### TAGS 参数（`ags/guidance_schedule.py`）

| 参数 | 值 | 说明 |
|:-:|:-:|:-:|
| `w_max` | 0.3 | 最大权重（被 CAGS per-class 值覆盖） |
| `schedule_type` | cosine | 调度类型 |
| `reverse` | True(high_noise) | high_noise 最强在 t_stop 端 |

权重公式: `w(t) = w_max · cos(π/2 · progress)` where progress 从最强端=0 到最弱端=1

### 4.3 引导窗口说明

- **high_noise**: 引导在去噪过程**前段**（高噪声 timestep, t ∈ [t_stop, t_max]）。MGD³ 论文使用此设置（`stop_t=25`，引导 t=25~49 共 25 步）
- **low_noise**: 引导在去噪过程**后段**（低噪声 timestep, t ∈ [0, t_stop]）

---

## 5. 实验机环境

| 项目 | 值 |
|:-:|:-:|
| GPU | 双 RTX 5090 (32GB each) |
| Python | 3.10.12 |
| PyTorch | 2.11.0+cu128 |
| 命令 | `python3`（非 `python`） |
| 数据路径 | `/root/data/imagenet100/`（100类, 126,689 train + 5,000 val） |
| DiT checkpoint | `/root/ICLR2027_KD/pretrained_models/DiT-XL-2-256x256.pt` |
| MGD³ 官方代码 | `/root/mode_guidance/`（已克隆） |
| 项目路径 | `/root/ICLR2027_KD/` |
| val 目录名 | `val`（非 `test`） |

### 实验机上的已有资源

- MGD³ 官方生成图像: `/root/mode_guidance/results/unguided/dataset_0/` 和 `/root/mode_guidance/results/guided/dataset_0/`
- MGD³ 官方评估结果: unguided=37.70%, guided=39.70%
- 修复后 100 类生成图像:
  - `/root/ICLR2027_KD/results/sweep_in100/high_noise_mgd3_fixed_d25/dataset_0/`（1000 张，已评估 10.64%）
  - `/root/ICLR2027_KD/results/sweep_in100/high_noise_cags_fixed_d25/dataset_0/`（1000 张，已评估 9.91%）
  - `/root/ICLR2027_KD/results/sweep_in100/high_noise_unguided_baseline_d25/dataset_0/`（1000 张，已评估 9.70%）
- 修复后 10 类生成图像:
  - `/root/ICLR2027_KD/results/sweep_in10/high_noise_mgd3_d25_fixed_d25/dataset_0/`（100 张，已评估 39.00%）
- cluster_cache.pkl: `/root/ICLR2027_KD/results/sweep_in100/cluster_cache.pkl` 和 `sweep_in10/cluster_cache.pkl`

---

## 6. 实验复现命令

### 6.1 快速评估已有图像（推荐，~95s/seed/1000ep）

```bash
# 10 类
CUDA_VISIBLE_DEVICES=0 python3 quick_eval.py \
  --train-dir /path/to/generated/dataset_0 \
  --val-dir /root/data/imagenet100/val \
  --class-file ./misc/class100.txt \
  --nclass 10 --epochs 1000 --depth 6 --seeds 0 1

# 100 类
CUDA_VISIBLE_DEVICES=0 python3 quick_eval.py \
  --train-dir /path/to/generated/dataset_0 \
  --val-dir /root/data/imagenet100/val \
  --class-file ./misc/class100.txt \
  --nclass 100 --epochs 1000 --depth 6 --seeds 0 1
```

### 6.2 生成 + 评估（duration_sweep.py，较慢）

```bash
# MGD³-exact (固定 λ=0.1, stop_t=25)
CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
  --window high_noise --durations 25 --schedule constant \
  --spec imagenet100 --nclass 10 --ipc 10 \
  --imagenet-dir /root/data/imagenet100/ \
  --save-base ./results/sweep_in10 \
  --epochs 1000 --depth 6 \
  --no-cags --fixed-scale 0.1 --fixed-stop-t 25 \
  --seeds 0 1 --num-datasets 1 \
  --tag mgd3_fixed

# CAGS adaptive (自适应引导强度)
CUDA_VISIBLE_DEVICES=1 python3 duration_sweep.py \
  --window high_noise --durations 25 --schedule constant \
  --spec imagenet100 --nclass 10 --ipc 10 \
  --imagenet-dir /root/data/imagenet100/ \
  --save-base ./results/sweep_in10 \
  --epochs 1000 --depth 6 \
  --seeds 0 1 --num-datasets 1 \
  --tag cags_fixed

# Unguided baseline (λ=0)
CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
  --window high_noise --durations 25 --schedule constant \
  --spec imagenet100 --nclass 10 --ipc 10 \
  --imagenet-dir /root/data/imagenet100/ \
  --save-base ./results/sweep_in10 \
  --epochs 1000 --depth 6 \
  --no-cags --fixed-scale 0.0 --fixed-stop-t 0 \
  --seeds 0 --num-datasets 1 \
  --tag unguided_baseline
```

### 6.3 MGD³ 官方代码（对照实验）

```bash
cd /root/mode_guidance

# 生成
python3 sample_mode_guidance.py --model DiT-XL/2 --image-size 256 \
  --save-dir results/guided --spec imagenet100 --num-samples 10 \
  --guidance --stop_t 25 --imagenet_dir /root/data/imagenet100 --seed 0 --num-datasets 1

# 评估
python3 train.py -d imagenet --imagenet_dir results/guided/dataset_0 /root/data/imagenet100 \
  -n convnet --depth 6 --nclass 10 --norm_type instance --ipc 10 \
  --tag test --slct_type random --spec imagenet100 --repeat 1
```

---

## 7. 下一步计划（按优先级）

### P0: CAGS 参数调优（当前最大瓶颈）

CAGS 自适应引导强度（0.14-0.19）高于 MGD³ 固定 0.1，在 100 类下反而更差。需要：

1. **降低 `guidance_scale_range` 上限**: 从 `(0.05, 0.3)` 调整为 `(0.05, 0.15)` 或 `(0.05, 0.12)`，使 CAGS 输出更接近 0.1
2. **调整 sigmoid 中心和斜率**: 当前 `sigmoid(5 * (complexity - 0.5))`，可改为 `sigmoid(3 * (complexity - 0.6))` 降低整体强度
3. **在 10 类上测试 CAGS vs fixed**：10 类有 MGD³ 官方基准（39.7%），更容易判断 CAGS 是否有效
4. **搜索最优固定 λ**：测试 λ ∈ {0.05, 0.08, 0.1, 0.12, 0.15}，找到各设置下的最优值

### P1: 补全实验矩阵

| 实验 | 设置 | 状态 |
|:-:|:-:|:-:|
| 10 类 CAGS vs fixed | nclass=10, 1000ep | ❌ 待跑 |
| 10 类 unguided baseline | nclass=10, 1000ep | ❌ 待跑 |
| 100 类 CAGS (调参后) | nclass=100, 1000ep | ❌ 待跑 |
| 2000 epochs 最终数字 | nclass=10, 2000ep | ❌ 待跑 |
| 3 seeds 统计显著性 | 所有配置, 3 seeds | ❌ 待跑 |
| ImageNette / ImageWoof | nette/woof spec | ❌ 待跑 |
| CIFAR-10 / CIFAR-100 | 需适配 | ❌ 待跑 |

### P2: 扩展实验

1. **跨架构评估**: 用 ResNet-18, ResNet-50, ConvNet 不同 depth 评估同一蒸馏数据集
2. **FID 计算**: 实现 `evaluation/metrics.py` 中的 FID 计算
3. **不同 IPC**: IPC=1, 10, 50 对比
4. **不同引导窗口**: low_noise vs high_noise 系统对比
5. **TAGS 调度类型消融**: cosine vs linear vs exponential vs constant

### P3: 论文撰写

1. 搭建 LaTeX 骨架（main.tex + sections/）
2. 撰写 Method 部分（CAGS, IAST, TAGS 的数学定义和公式推导）
3. 撰写 Experiments 部分（主实验、消融、可视化）
4. 撰写 Related Work（区分与 MGD³、DATM、SRe²L 等的区别）

---

## 8. 关键技术细节

### 8.1 MGD³ 引导机制

MGD³ 在 DDIM 采样过程中，将预测的 `x_start` 引向真实图像的 mode features（通过 K-means 聚类中心获取）：

1. 用 VAE encoder 提取训练集每类图像的 latent features
2. 对每类做 K-means 聚类（K=IPC），取聚类中心作为 mode features
3. 在采样过程中，在 `stop_t` 之前/之后的步骤中，将预测的 `x_start` 拉向 mode features
4. 引导公式: `img = mean + (-(x_start - mode) * w * exp(0.5*log_var)) + noise`

### 8.2 CAGS 复杂度计算

```
1. 对每类 latent features 做 K-means (K 范围 2-20，用 silhouette score 选最优 K)
2. 计算 mode count score = K / max_K
3. 计算 intra-class entropy = -Σ(p_k * log(p_k)) / log(K)  (归一化)
4. complexity = sigmoid(5 * (α * mode_count_score + β * entropy - 0.5))
5. guidance_strength = min_s + complexity * (max_s - min_s)
```

### 8.3 IAST 停止时机

```
guidance_steps = t_max * (1 - exp(-λ * sqrt(IPC) / K(c))) + complexity_weight * complexity * t_max
t_stop = guidance_steps (low_noise) or t_max - 1 - guidance_steps (high_noise)
```

### 8.4 quick_eval.py vs duration_sweep.py 评估

- `quick_eval.py`: 将所有图像预加载到 GPU，在 GPU 上做数据增强，~95s/seed/1000ep（10 类），~950s/seed/1000ep（100 类）。**推荐使用**
- `duration_sweep.py`: 用 DataLoader，慢得多（25+ min/1000ep），但支持更多参数
- 两者评估设置完全一致: ConvNet-6, CutMix, RandomResizedCrop, MultiStepLR, batch_size=64

### 8.5 注意事项

1. **MGD³ 用 nclass=10**: `imagenet100.sh` 脚本用 `--nclass 10`，只取 class100.txt 前 10 个类
2. **val 目录名**: ImageNet-100 的验证集目录名为 `val`（非 `test`）
3. **cluster_cache.pkl 可复用**: 特征提取耗时，缓存后无需重新计算（但换 nclass 需重新生成）
4. **同时运行 2 个 quick_eval 会拖慢图像加载**: 建议串行加载或间隔 2 分钟启动
5. **PYTHONUNBUFFERED=1**: 运行时加此环境变量以实时查看输出

---

## 9. 已知问题与风险

1. **CAGS 当前不如 fixed**: 自适应强度过高，需要调参（P0 优先级）
2. **100 类结果绝对值低**: 10.64% 看似低，但这是 IPC=10 + 1000 epochs 的结果；MGD³ 用 2000 epochs
3. **缺乏统计显著性**: 当前多数实验仅 2 seeds，需要 3+ seeds 和显著性检验
4. **FID 未实现**: `evaluation/metrics.py` 仅有占位代码
5. **跨架构未测试**: `evaluation/cross_arch.py` 从未运行
6. **CIFAR 未适配**: 需要修改数据加载和采样代码适配 CIFAR 分辨率
