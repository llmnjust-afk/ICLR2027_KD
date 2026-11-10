MGD³ 单项同协议对照：RTX 4090 专用入口

默认最多两张 4090，从登录节点执行：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_mgd3_4090.sh

只用一张卡：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_mgd3_4090.sh --max-parallel 1

查看提交记录与完成数量：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_mgd3_4090.sh --status

预览而不提交：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_mgd3_4090.sh --dry-run

只新增一个方法条件
  MGD³ 模式引导系数 0.1；stop_t=25；50 步采样中 t>25 的 24 步启用模式引导。
  生成种子 100–104：5 份数据；训练种子 0、1、2：15 次 ConvNet6 训练。
  ImageNet100、IPC=10、DiT-XL/2、固定 CFG=4、batch_gen=4、batch_train=128。
  每次正式训练 2000 epoch、FP32、固定最终检查点，沿用核心实验的原始验证集。
  不运行 β-CFG、特征提取器敏感性、其他聚类实验或新超参数搜索。

基准复用
  从已完成 results/rebuttal_core_v2/runs/final 复用 CAGS、fixed_mean、no_mode。
  共 45 次训练结果；保留原结果路径、SHA-256、原 plan_hash 及生成清单。
  不重新训练基准、不复制已有图像或分类器权重、不改写原文件。
  新输出中的基准 JSON 是带来源标记的导入记录；其中 checkpoint_sha256
  指向原结果相邻的 .pt 文件，新目录不另造这些权重。
  使用核心实验的完整冻结代码快照；独立锁管理器使用已修复的 BeeGFS 锁逻辑。
  所有生成种子、分类器种子、图像和训练协议保持配对；只允许与基准同型号 RTX4090。
  导入全部预设重复，不按准确率挑选种子或样本。

排队流程
  GPU 自检 -> 5 个生成任务 -> 15 个分类器训练任务 -> CPU 汇总。
  每个 GPU 任务申请一张 gpu:4090:1，rtx4090 分区，8 CPU、32GB 主机内存。
  各 GPU 数组默认并行上限 2，通过依赖关系保证整批最多同时使用两张卡。
  自检包含精确停止窗口的生成，以及真实 100 类规模、batch=128 的两轮分类器训练。
  自检结果单独存放，绝不进入论文准确率。
  CPU 汇总申请 2 CPU、8GB 主机内存。
  使用项目 ICLR/bin/python，不需要手动 conda activate；计算节点不需要下载依赖或模型。
  默认 account=general；如账号另有分配，可在首次提交前设置 CAGS_4090_ACCOUNT。
  默认沿用账号 QOS，需要指定时可设置 CAGS_4090_QOS。

重复提交和故障恢复
  重复执行会显示已提交状态，不会再次创建同一批作业。
  首次 --max-parallel 1 的选择会被保存，之后省略参数仍使用一张卡。
  各阶段正式提交前都会使用 sbatch --test-only 检查资源请求。
  部分阶段被明确拒绝后，重复命令会补交缺失阶段，不重复提交已成功的阶段。
  提交超时或返回无法识别时标为 unknown；必须核对 Slurm，禁止盲目重投。
  已知失败且所有旧作业均终止后，可加 --retry-failed；已完成数据和结果经校验后跳过。
  如果仍有后续依赖作业处于队列中，不能越过终止检查启动重叠批次。
  不擅自删除归属不明的锁；不修改已有 A800 或 4090 任务。

输出
  experiments/rebuttal_mgd3_4090_v1/submission.json：提交成功后的 Slurm ID。
  results/rebuttal_mgd3_4090_v1/slurm/：自检、生成、训练、汇总日志。
  results/rebuttal_mgd3_4090_v1/runs/final/summary.txt：完整配对结果。
  results/rebuttal_mgd3_4090_v1/runs/final/mgd3_comparison.json：审稿问题对应的 CAGS−MGD³ 比较。
  results/rebuttal_mgd3_4090_v1/smoke_tests/：自检输出，不用于论文。

方法与证据范围
  官方采样更新和停止窗口核对来源：
    https://github.com/jachansantiago/mode_guidance/blob/main/diffusion/gaussian_diffusion.py
    https://github.com/jachansantiago/mode_guidance/blob/main/sample_mode_guidance.py
    https://github.com/jachansantiago/mode_guidance/blob/main/scripts/nette.sh
  参考文件保存在 reference/，其具体内容由计划中的 SHA-256 固定。
  已有 CPU 逐步数值测试对照官方实现，而不是仅依据论文描述猜测公式。
  数据划分、中心裁剪/VAE 均值特征、K=IPC 模式目标、配对 RNG、批处理及分类器
  使用本项目共同协议。因此结果应描述为 MGD³ 官方采样更新与停止窗口在共同协议
  下的对照，不能称为复现原论文整张结果表。
  CAGS 保留原 25 个引导步骤；MGD³ 精确官方停止窗口有 24 个引导步骤，此差异明确记录。

运行时间是估计，不是结束保证
  依据已有 4090 耗时：生成每份约12分钟，ConvNet6 每次训练约47–48分钟。
  一张卡约13–15小时，两张卡约7–9小时；排队、异常重跑及共享节点波动另计。
  本入口在开发环境完成 CPU/提交模拟验证，真实 GPU 自检由首个已分配作业执行。
