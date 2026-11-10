两张 A800：补齐跨架构核心对照

在登录节点运行（无需额外 conda activate）：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_a800.sh

只预览，不提交：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_a800.sh --dry-run

查看结果数量和 A800 实测速度：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_a800.sh --status

失败批次全部终止后，保留同型号 A800 上已完成的训练并重试：
  bash /home/lulm/ICLR2027_KD-main/scripts/submit_rebuttal_a800.sh --retry-failed

实验范围
- ResNet-18 与 ResNetAP-10；ImageNet-100；IPC=10。
- 六组：CAGS、固定均值、关闭模式引导、三组类别系数置换。
- 5 个生成种子 × 3 个训练种子 × 2 个架构 × 6 个条件，共180次训练。
- 复用旧跨架构批次中8个完整配对块，即48次RTX 4090训练的结果与权重。
- 其余22个配对块、132次训练在A800执行。旧批次中未完成块内已有7次完整训练，
  这7次结果保留在旧目录，但不纳入新汇总；需要在A800重新训练，以保持六组方法硬件匹配。
- 同一配对块指同一架构、生成种子和训练种子。其全部六个条件使用同一GPU型号。
- 使用旧批次固定的生成图片，不重新生成图片；不以已有准确率选择迁移任务。
- 新目录 results/rebuttal_crossarch_a800_v1；旧跨架构目录和当前4090调参目录不修改。
- 最终汇总明确披露不同配对块使用不同GPU型号；不声称这是一套全部由A800运行的实验。

协议与速度
- 2000 epochs、batch=128、FP32、原TF32设置、原数据增强、固定末轮评估均不改变。
- 每个新配对块先执行20轮显存与计时检查；诊断不纳入180次正式训练统计。
- 每次正式训练重新设置随机种子，诊断不改变正式实验的初始化与随机流。
- SMOKE_PASSED.json记录显存、实际GPU型号及预热后的每轮时间；完整训练结果记录elapsed_s。
- 不承诺A800单卡比4090更快。原4090上两种架构每次约17.3/22.7分钟；
  若A800速度相当，两卡完成132次训练约22–25小时，不含排队。以实际计时修正估计。

调度
- 确认本集群分区为a800、GRES为gpu:a800:1；数组并发上限为2。
- 默认查询当前用户的Slurm账户和已授权A800 QOS，不自动选择urgent。
- 必要时可显式设置 CAGS_A800_QOS 与 CAGS_A800_ACCOUNT；值必须来自自己的实际分配。
- 提交前核实旧GPU数组157933已全部终止，避免与旧入口重复训练。
- 提交前对GPU和CPU汇总请求都执行sbatch --test-only。
- 单独CPU汇总作业申请cpu分区，不占用第三张GPU。
- 重复执行普通提交命令复用作业号；sbatch返回不明确时禁止盲目重提。
- 当前工具环境无法访问Slurm控制器，未替用户提交GPU作业，也未验证真实A800运行。

来源与验证
- 评估使用 experiments/rebuttal_crossarch_v1/code 中的冻结代码，每次核验文件哈希。
- 新导入的结果记录保留原测量、原计划哈希和原JSON路径及哈希，检查点链接至原文件。
- 本目录frozen_manifest.json与结果plan.json记录完整迁移清单及启动器哈希。
- 完整结果：results/rebuttal_crossarch_a800_v1/runs/final/summary.txt、summary.json、
  raw_results.csv、reviewer_evidence.csv；原始结果CSV包含GPU型号。
- 测试：ICLR/bin/python experiments/rebuttal_crossarch_a800_v1/test_launcher.py

本入口只补齐跨架构核心对照，不包括外部自适应CFG方法、公开MGD3忠实复现或估计器敏感性实验。
