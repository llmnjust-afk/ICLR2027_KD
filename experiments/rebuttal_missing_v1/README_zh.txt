缺失实验补充批次（2026-11-10）

提交全部任务：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_missing.sh

只使用一张 A800 串行排队（首次提交时选择）：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_missing.sh --max-parallel 1

查看提交记录和结果数量：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_missing.sh --status

预览（不会提交）：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_missing.sh --dry-run

资源与流程
  每个 GPU 任务只申请一张 A800；默认最多并行两项，因此本批最多同时使用两张。
  使用现有 a800 分区/QOS，参考已有 A800 80GB 作业配置；不需要多卡分布式训练。
  GPU 任务各申请 8 CPU、32GB 主机内存；主机内存与 GPU 显存是不同资源。
  CPU 聚类任务各申请 8 CPU、32GB 主机内存，最多两项并行；汇总只需 2 CPU、8GB。
  脚本使用项目 ICLR/bin/python，无需手动 conda activate，也无需在登录节点运行 CUDA。
  所有模型和权重都已在本机；计算节点无需联网或安装软件。
  是否立即启动由 Slurm 资源/账号配额决定；不承诺 A800 比 4090 更快。

一条命令提交七个作业/数组入口
  1. CPU：立刻分析已有 core、tune、final 的逐类别结果（未完成的明确标为 partial）。
  2. 单卡：GPU 生成/分类器自检、β-CFG 适配自检，然后提取 ResNet-18 特征。
  3. CPU 数组：五种复杂度估计变体。
  4. CPU：冻结完整实验计划和各类别引导系数。
  5. 单卡数组：90 份合成数据集，最多两项并行。
  6. 单卡数组：15 个配对训练块，各含 18 个条件，共 270 次分类器训练，最多两项并行。
  7. CPU：统计与逐类别分析，并刷新现有实验的分析。
  第 2–7 步按成功依赖顺序执行；CPU 阶段不占 GPU。

18 个条件的具体组成
  5 个基准：no_mode、fixed_mean、fixed_best_convnet6、cags、cags_selected_convnet6。
  1 个 MGD³：官方更新公式，lambda=0.1，精确 stop_t=25，在 50 步中引导 24 步。
  2 个 β-CFG：单独使用，以及与默认 CAGS 组合。
  10 个敏感性条件：五种估计器各有原始系数和均值匹配系数。
    五种估计器为 VAE K=5、VAE K=10、VAE PCA=16 自动 K、VAE Ward K=10、ResNet18 自动 K。
    Ward 与 KMeans K=10 的比较固定 K/PCA，仅改变聚类方法。
    生成时的模式目标全部固定为原准备缓存；不重新提取原 VAE 特征。

协议与重复实验
  ImageNet-100、IPC=10、DiT-XL/2、50 步、ConvNet6、2000 epoch、FP32。
  新生成种子 300–304，训练种子 0/1/2；相同生成/类别/块使用配对初始噪声和逐步噪声。
  使用原数据划分、最终验证集、分类器规则及已经冻结的留出集选参结果。
  5 个基准使用新种子并在本批同型号 A800 上运行，避免新旧硬件/随机样本混杂；
  这是新增方法所需的配对基准，不会重启、覆盖现有作业或把旧结果混入本批统计。
  重复执行提交命令只显示已记录的作业，不重复创建队列。
  后续文件变更会被冻结哈希检查拦截，现有主程序和正在运行的源码保持原样。

公开方法的可报告范围
  MGD³ 核对来源：
    https://github.com/jachansantiago/mode_guidance/blob/main/diffusion/gaussian_diffusion.py
    https://github.com/jachansantiago/mode_guidance/blob/main/sample_mode_guidance.py
    https://github.com/jachansantiago/mode_guidance/blob/main/scripts/nette.sh
  采用官方模式引导更新和 stop_t=25；CPU 测试验证完整轨迹和停止边界。
  数据划分、图像预处理、批处理、噪声配对及评价器采用本项目共同协议。
  因而应写“官方 MGD³ 采样更新在共同协议下的比较”，不能称复现论文表格精度。

  β-CFG 核对来源：
    https://github.com/gmum/beta-CFG/blob/main/latent_diffusion.py
    https://github.com/gmum/beta-CFG/blob/main/README.md
  采用官方示例参数 scale=7.5、a=2、b=3、gamma=1，包括梯度范数归一化和未做峰值归一化的 Beta PDF。
  原方法的 SD/DDIM 被替换为 DiT/DDPM，共同保留 DiT 原有的前三通道 CFG 约定。
  参数在观察本批结果前固定，不使用最终验证集选参。
  这是一项明确披露的 DiT 适配对照，并非原 SD 论文指标的完整复现，
  也不能据此宣称胜过经过充分调参的 β-CFG；若审稿人要求原任务完整复现，仍需另外设计。
  原始参考文件已保存于 reference/，通过 frozen_manifest.json 中的 SHA-256 固定具体内容。

逐类别分析
  复用已保存的每类正确数和总数，训练种子先在每个生成种子内求平均。
  输出复杂度/熵/离散度/可分性与分类准确率、配对增益的 Pearson/Spearman 相关。
  输出动物与其他类别的增益及生成级配对区间；其他类别不一律称为“物体”。
  对已有 tune 固定系数网格（包含零），另外做留一生成种子选择系数的描述性分析。
  不把相关性当作因果性、不把离散网格最优点当作真正最优系数，也不据此证明单调性定理。

输出位置
  experiments/rebuttal_missing_v1/submission.json：实际 Slurm 作业编号（提交成功后生成）。
  results/rebuttal_missing_v1/slurm/：按阶段区分的运行日志。
  results/rebuttal_missing_v1/runs/final/：计划、合成数据、训练结果与汇总。
  results/rebuttal_missing_v1/analysis/：逐类别 CSV、相关性、分组和留一生成分析。
  results/rebuttal_missing_v1/smoke_tests/：自检；不得用于论文结果。

故障恢复
  如果有失败，先查看 --status 对应的 Slurm 状态和阶段日志。
  所有上次提交的作业均已结束后，修复外部故障可用 --retry-failed 重排；
  已完成特征、数据集和分类器结果会经过身份/校验检查后跳过。
  若第一次使用 --max-parallel 1，恢复时也传入 --max-parallel 1。
  仍有活跃作业、锁归属不明、提交返回未知状态时，脚本不会擅自删锁或重复提交。
  修改科学协议或冻结代码需建立新版本批次，不能改动本批继续混跑。

验证记录
  9 项 CPU 测试通过：官方公式/轨迹对照、DiT 50 步适配、冻结计划、
  Slurm 成功/异常状态记录、依赖顺序和重复提交保护。
  Bash 语法和 Python 编译检查通过；已有 core/tune 逐类别分析试跑通过。
  当前工具环境无法访问 Slurm 控制器，未提交真实作业，也未声称完成真实 A800 自检。
