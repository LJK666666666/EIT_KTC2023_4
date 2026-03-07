 KTC2023 训练框架实现计划                                                                                                                                             
  
 背景

 推理评估管线已完成（FCUNet/PostP/CondD三种方法均可正常推理）。现在需要将原始训练代码（位于 KTC2023_SubmissionFiles/ktc_training/）重构为可扩展的训练框架，集成到现有 
  src/ 代码结构中，以便后续研究新的重建方法。

 架构设计

 采用 BaseTrainer 抽象基类 模式，与现有 BasePipeline 对称：pipeline 负责推理，trainer 负责训练。三个训练器子类分别实现 FCUNet/PostP/CondD 的训练逻辑。

 src/
 ├── trainers/                        # 新增：训练框架
 │   ├── __init__.py
 │   ├── base_trainer.py              # BaseTrainer ABC
 │   ├── fcunet_trainer.py            # FCUNetTrainer（两阶段训练）
 │   ├── postp_trainer.py             # PostPTrainer（交叉熵分类）
 │   └── condd_trainer.py             # CondDTrainer（扩散模型+EMA）
 ├── data/
 │   ├── sim_dataset.py               # 新增：训练数据集类
 │   └── phantom_generator.py         # 新增：合成Phantom生成
 ├── configs/                         # 修改：补充训练超参数
 │   ├── base_config.py               # 修改：共用训练配置字段
 │   ├── fcunet_config.py             # 修改：补充训练参数
 │   ├── postp_config.py              # 修改：补充训练参数
 │   └── condd_config.py              # 修改：补充resume/data字段
 scripts/
 ├── train.py                         # 新增：统一训练入口
 └── generate_data.py                 # 新增：训练数据生成

 实现步骤

 步骤 1：数据集类（src/data/）

 src/data/phantom_generator.py
 - 提取自 KTC2023_SubmissionFiles/ktc_training/src/dataset/SimDataset.py:211-394
 - 函数：create_phantoms(), generate_polygon(), random_angle_steps()
 - 生成 256×256 随机phantom图像（{0,1,2} 三类分割）

 src/data/sim_dataset.py
 - FCUNetTrainingData(Uref, InvLn, base_path) — FCUNet用：加载 gt + raw measurements，运行时添加参考噪声
 - SimData(level, base_path, one_hot_gt=False) — PostP/CondD用：加载 gt + 5通道初始重建
 - MmapDataset(level, num_samples, base_path, one_hot_gt=False) — 大数据集的内存映射版本
 - 参考源：SimDataset.py 的 TrainingData(L104-138), SimData(L146-175), MmapDataset(L177-207)

 步骤 2：配置增强（src/configs/）

 src/configs/base_config.py — 添加 get_base_training_config() 返回共用训练字段：
 - resume_from, max_iters, num_workers, val_freq, save_freq

 src/configs/fcunet_config.py — 补充训练参数：
 - init_epochs=15, lr=3e-5, init_lr=1e-4, scheduler_step_size=30, scheduler_gamma=0.95
 - epochs=500, batch_size=6, grad_clip_norm=1.0
 - 数据路径：dataset_base_path, ref_path, mesh_name

 src/configs/postp_config.py — 补充训练参数：
 - lr=3e-5, scheduler_step_size=10, scheduler_gamma=0.75
 - epochs=500, batch_size=6, grad_clip_norm=1.0
 - level_to_num 映射每级样本数

 src/configs/condd_config.py — 补充：
 - resume_from=None, dataset_base_path, level_to_num
 - 已有 ema_decay/ema_warm_start_steps/epochs 等

 步骤 3：BaseTrainer（src/trainers/base_trainer.py）

 核心抽象基类，提供：
 - 自动递增结果目录：results/{experiment_name}_{num}/
 - 检查点：每 epoch 结束保存 last.pt（含model/optimizer/scheduler/epoch/global_step/best_metric/training_log + 子类额外状态）；验证分数提升时保存 best.pt
 - 恢复训练：从 last.pt 恢复所有状态，包括子类额外状态（如EMA）
 - 训练日志：training_log.json 每 epoch 更新（lr/loss/val_score等）
 - TensorBoard：自动记录 loss/lr/验证指标
 - 快速测试：max_iters 参数，训练指定迭代次数后停止
 - 进度条：tqdm

 4个抽象方法（子类必须实现）：
 def build_model(self):        # 创建 model/optimizer/scheduler
 def build_datasets(self):     # 创建 train_loader/val_data
 def train_step(self, batch):  # 单batch训练，返回 {'loss': float}
 def validate(self, epoch):    # 验证，返回 {'score': float}

 4个可选钩子：
 def on_epoch_start(self, epoch):          # epoch开始前
 def on_epoch_end(self, epoch, metrics):   # epoch结束后
 def get_checkpoint_extra(self):           # 额外检查点状态（如EMA）
 def load_checkpoint_extra(self, state):   # 恢复额外状态

 步骤 4：FCUNetTrainer（src/trainers/fcunet_trainer.py）

 最复杂的训练器，需要两阶段训练：

 阶段1（15 epochs）：仅训练 initial_linear 层
 - Optimizer: Adam(model.initial_linear.parameters(), lr=1e-4)
 - Loss: MSE（线性层输出 vs GT差值图）
 - 每batch随机采样 level 1-7，用 vincl 掩码将无效测量置零

 阶段2（500 epochs）：训练完整模型
 - Optimizer: Adam(model.parameters(), lr=3e-5)
 - Scheduler: StepLR(step_size=30, gamma=0.95)
 - Loss: CrossEntropyLoss
 - 同样的 level 增强 + vincl 掩码
 - 梯度裁剪 norm=1.0

 验证：每 epoch 用 4 个挑战测试图像 × 7 级别，计算 FastScoringFunction，跟踪 best_score
 检查点额外状态：training_stage, init_epoch_count, init_optimizer_state_dict

 关键参考：KTC2023_SubmissionFiles/ktc_training/train_FCUNet.py

 步骤 5：PostPTrainer（src/trainers/postp_trainer.py）

 最简单的训练器（单阶段，无EMA）：

 - 数据：ConcatDataset 合并 7 个 level 的 MmapDataset（共~116K样本）
 - 模型：OpenAiUNetModel(in_channels=5, out_channels=3, marginal_prob_std=None)
 - Optimizer: Adam(lr=3e-5), Scheduler: StepLR(step_size=10, gamma=0.75)
 - Loss: CrossEntropyLoss（train_step 中将 GT 转为 one-hot 编码）
 - 验证：用预计算的 ChallengeReconstructions + FastScoringFunction

 关键参考：KTC2023_SubmissionFiles/ktc_training/train_postprocessing.py

 步骤 6：CondDTrainer（src/trainers/condd_trainer.py）

 替代现有 src/diffusion/trainer.py 中的函数式训练器：

 - 每级别独立训练：CondDTrainer(config, level=3) 训练单个 level 的模型
 - 模型：get_standard_score() + get_standard_sde() from src/diffusion/exp_utils
 - Loss：根据 SDE 类型自动选择 epsilon_based_loss_fn（DDPM）或 score_based_loss_fn
 - EMA：decay=0.999，warm_start=50步后开始更新
 - Optimizer: Adam(lr=1e-4)，无 scheduler
 - 验证：DDIM采样4张图 → round → FastScoringFunction
 - 检查点额外状态：ema_state_dict, ema_initialized

 关键参考：KTC2023_SubmissionFiles/ktc_training/train_score.py

 步骤 7：统一训练入口（scripts/train.py）

 # 训练 FCUNet
 python scripts/train.py --method fcunet

 # 训练 PostP
 python scripts/train.py --method postp

 # 训练 CondD level 3
 python scripts/train.py --method condd --level 3

 # 快速测试（1次迭代）
 python scripts/train.py --method fcunet --max-iters 1

 # 恢复训练
 python scripts/train.py --method fcunet --resume results/fcunet_baseline_1/last.pt

 # 覆盖超参
 python scripts/train.py --method postp --epochs 100 --lr 1e-4 --batch-size 4

 步骤 8：训练数据生成（scripts/generate_data.py）

 # 为 level 3 生成 2000 个训练样本
 python scripts/generate_data.py --level 3 --num-images 2000

 # 为所有 level 生成
 python scripts/generate_data.py --all-levels --num-images 2000

 流程：随机phantom → EITFEM正演模拟（带噪声）→ 5组线性化重建 → 保存 gt/measurements/recos

 步骤 9：验证

 1. python scripts/train.py --method fcunet --max-iters 1 — 快速测试 FCUNet
 2. python scripts/train.py --method postp --max-iters 1 — 快速测试 PostP
 3. python scripts/train.py --method condd --level 1 --max-iters 1 — 快速测试 CondD
 4. 检查 results/ 下生成的目录、last.pt、training_log.json、TensorBoard
 5. 测试恢复训练：--resume results/.../last.pt

 扩展新方法的流程

 未来添加新重建方法只需：
 1. 在 src/configs/ 添加配置文件
 2. 在 src/trainers/ 添加 Trainer 子类（实现4个抽象方法）
 3. 在 scripts/train.py 添加一个 elif 分支
 4. （可选）在 src/data/sim_dataset.py 添加数据集类
 5. （可选）在 src/pipelines/ 添加推理管线

 关键源文件参考

 ┌────────────────┬────────────────────────────────────────────────────────────────┐
 │      用途      │                             源文件                             │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ FCUNet训练逻辑 │ KTC2023_SubmissionFiles/ktc_training/train_FCUNet.py           │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ PostP训练逻辑  │ KTC2023_SubmissionFiles/ktc_training/train_postprocessing.py   │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ CondD训练逻辑  │ KTC2023_SubmissionFiles/ktc_training/train_score.py            │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 数据集类       │ KTC2023_SubmissionFiles/ktc_training/src/dataset/SimDataset.py │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 数据生成       │ KTC2023_SubmissionFiles/ktc_training/create_training_data.py   │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 扩散训练器     │ src/diffusion/trainer.py（已有，CondDTrainer将替代）           │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 扩散损失函数   │ src/diffusion/losses.py（已有，直接复用）                      │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ EMA            │ src/diffusion/ema.py（已有，直接复用）                         │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ SDE            │ src/diffusion/sde.py（已有，直接复用）                         │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 模型构建工具   │ src/diffusion/exp_utils.py（已有，直接复用）                   │
 ├────────────────┼────────────────────────────────────────────────────────────────┤
 │ 评分函数       │ src/evaluation/scoring.py（已有，验证时复用）                  │
 └────────────────┴────────────────────────────────────────────────────────────────┘
