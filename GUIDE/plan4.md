 SAE (Sparse AutoEncoder) 实现计划（最终版 v3）                                                                                                               
                                                                                                                                                              
 Context                                                                                                                                                      
  
 前两种方法逐像素编码过于离散，与电导率图像低频特征不符。SAE 将问题解耦为两阶段：
 1. ST-SAE 学习 GT 图像的紧凑表示（65维 = 63维形态 + 2维角度 [cosθ, sinθ]）
 2. 纯 MLP 预测器：电极测量 → 隐向量 → 冻结解码器 → 重建图像

 架构设计

 ST-SAE 模型 (src/models/sae/model.py)

 输入: One-hot GT (B, 3, 256, 256)

 正向传播:
 1. 角度预测: 原始图像 → AngleCNN → 2维 → F.normalize → angle_xy = [cosθ, sinθ]
   - AngleCNN 接收原始图像，避免循环依赖
   - θ = atan2(sinθ, cosθ) 仅用于 STN 旋转
 2. 逆向对齐: F.affine_grid + F.grid_sample 旋转 -θ
   - 训练时 mode='bilinear'：保证梯度能反传给 AngleCNN（nearest 梯度处处为零）
   - 推理时 mode='nearest'：保持 One-hot 离散性
 3. 纯形态压缩: 标准姿态图像 → EncoderCNN → z_shape (63维)
   - Conv: 3→32→64→128→256，每层 BN+LeakyReLU, stride=2
   - 4次下采样: 256×16×16 → flatten → FC → 63维
 4. 纯形态解码: 只把 63维 送入 Decoder → 标准姿态 logits (3,256,256)
   - 分步展开: FC(63, 256×4×4=4096) → reshape(256,4,4)
   - ConvTranspose 上采样 (6步): 256→256→128→64→32→16→8 → 1×1Conv(8,3)
   - 4→8→16→32→64→128→256，最后 1×1Conv 从 8→3（平滑过渡，避免 16→3 突变）
   - 最后无激活（raw logits）
 5. 正向归位: F.grid_sample(mode='bilinear') 旋转 +θ（logits 是连续值）

 隐向量: z = [z_shape(63), cosθ(1), sinθ(1)] = 65维

 损失函数 — CrossEntropy + L1 + 等变性约束

 target_indices = torch.argmax(gt_onehot, dim=1)  # (B, 256, 256)

 # 主重建损失
 recon_loss = F.cross_entropy(output_logits, target_indices)

 # 稀疏约束（只惩罚63维shape，绝不惩罚angle_xy）
 sparsity_loss = l1_lambda * torch.mean(torch.abs(z_shape))

 # 旋转等变性约束（确保z_shape对旋转不变）
 # 随机旋转输入k步，检查编码后z_shape是否一致
 k = random.randint(1, 31)
 image_rot = rotate_image(gt_onehot, k * 2*pi/32)
 z_shape_rot, _ = model.encode(image_rot)
 equiv_loss = equiv_lambda * F.mse_loss(z_shape_rot, z_shape)
 # 两个分支都不detach，互相约束，收敛更快

 total_loss = recon_loss + sparsity_loss + equiv_loss

 纯 MLP 预测器 (src/models/sae/predictor.py)

 class MeasurementPredictor(nn.Module):
     # backbone: 2356 → 512 → 256 → 128
     # 每层 Linear + LayerNorm + LeakyReLU + Dropout(0.1)
     # head_shape: FC(128, 63)
     # head_angle: FC(128, 2) → F.normalize

 Loss: MSE(pred_shape, target_shape) + λ_angle(=0.5) × MSE(pred_angle_xy, target_angle_xy)
 - λ_angle=0.5 补偿 63维 vs 2维 的量级差异

 旋转数据增强（Phase 3 DataLoader）

 if self.augment_rotation:
     k = np.random.randint(0, 32)
     # 电压循环移位: (2356,) → reshape(31, 76) → np.roll(axis=0, shift=k) → flatten
     # axis=0 对应31个差分电极通道（需确认与data.h5存储格式一致）
     measurements = measurements.reshape(31, 76)
     measurements = np.roll(measurements, shift=k, axis=0)
     measurements = measurements.flatten()
     # z_shape 不变；angle_xy 旋转 delta = k × 2π/32
     delta = k * (2 * np.pi / 32)
     cos_new = cos_old * np.cos(delta) - sin_old * np.sin(delta)
     sin_new = sin_old * np.cos(delta) + cos_old * np.sin(delta)

 训练流程

 Phase 1: SAE 训练 (python scripts/train.py --method sae)

 - 数据集: GTHDF5Dataset（只读 gt，返回 gt_onehot + gt_indices）
 - Loss = CE + l1_lambda×L1(z_shape) + equiv_lambda×L_equivariance
 - AdamW, lr=1e-3
 - ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-5)
 - 定期保存重建可视化
 - 训练后验证: 取100张图旋转0°/90°/180°/270°，检查z_shape偏差和角度跟随

 Phase 2: 编码并保存（训练结束自动执行）

 - model.eval() + torch.no_grad()：确保 BN 使用 running stats
 - 编码全部 GT → [z_shape(63), cosθ(1), sinθ(1)] = 65维
 - 统计 z_shape 稀疏阈值: threshold = np.percentile(np.abs(all_z_shapes), 5)
 - 保存至 {result_dir}/latent_codes.h5:
   - codes: (N, 65) float32
   - indices: (N,) int — 与 data.h5 索引对齐校验
   - sparsity_threshold: scalar float — z_shape 5%分位阈值，供推理时截断

 Phase 3: 预测器训练 (python scripts/train.py --method sae_predictor)

 - 数据集: SAEPredictorHDF5Dataset
   - 通过 indices 对齐 latent_codes.h5 和 data.h5
   - vincl 掩码 + 噪声增强 + 旋转增强（循环移位）
   - 返回 (measurements[2356], target_z[65])
 - Loss: MSE(shape) + 0.5 × MSE(angle_xy)
 - 冻结 SAE 解码器用于验证可视化
 - AdamW, lr=3e-4

 推理 (src/pipelines/sae_pipeline.py)

 1. 预处理测量值（减参考、vincl）
 2. MLP → (z_shape[63], angle_xy[2])
 3. Decoder(z_shape) → 标准姿态 logits (3,256,256)
 4. θ = atan2(sin, cos) → 旋转 +θ (bilinear) → argmax → 分割图 {0,1,2}

 推理时软阈值截断: 对 z_shape 做稀疏截断，抑制接近零的噪声维度
 z_shape = z_shape * (z_shape.abs() > threshold).float()
 # threshold = Phase2编码时统计的z_shape绝对值5%分位数，保存在checkpoint中

 文件清单

 新建 (8 files)

 ┌─────┬───────────────────────────────────────┬───────────────┐
 │  #  │                 文件                  │     说明      │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 1   │ src/models/sae/__init__.py            │ 导出          │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 2   │ src/models/sae/model.py               │ ST-SAE        │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 3   │ src/models/sae/predictor.py           │ 纯 MLP 预测器 │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 4   │ src/configs/sae_config.py             │ SAE 配置      │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 5   │ src/configs/sae_predictor_config.py   │ 预测器配置    │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 6   │ src/trainers/sae_trainer.py           │ SAE 训练      │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 7   │ src/trainers/sae_predictor_trainer.py │ 预测器训练    │
 ├─────┼───────────────────────────────────────┼───────────────┤
 │ 8   │ src/pipelines/sae_pipeline.py         │ 推理管线      │
 └─────┴───────────────────────────────────────┴───────────────┘

 修改 (7 files)

 ┌─────┬───────────────────────────┬──────────────────────────────────────────┐
 │  #  │           文件            │                   修改                   │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 9   │ src/data/sim_dataset.py   │ +GTHDF5Dataset, +SAEPredictorHDF5Dataset │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 10  │ src/data/__init__.py      │ 导出                                     │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 11  │ src/configs/__init__.py   │ 注册                                     │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 12  │ src/trainers/__init__.py  │ 注册                                     │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 13  │ src/pipelines/__init__.py │ 注册                                     │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 14  │ scripts/train.py          │ +sae, +sae_predictor                     │
 ├─────┼───────────────────────────┼──────────────────────────────────────────┤
 │ 15  │ scripts/evaluate_all.py   │ +sae                                     │
 └─────┴───────────────────────────┴──────────────────────────────────────────┘

 配置

 sae_config.py

 epochs=200, batch_size=32, lr=1e-3
 l1_lambda=1e-3, equiv_lambda=0.1
 shape_dim=63, encoder_channels=(32,64,128,256)
 decoder_start_size=4
 scheduler_patience=10, scheduler_factor=0.5, min_lr=1e-5

 sae_predictor_config.py

 epochs=300, batch_size=128, lr=3e-4
 sae_checkpoint='', latent_h5_path=''
 mlp_hidden_dims=(512,256,128), dropout=0.1
 lambda_angle=0.5
 finetune_decoder=False

 关键设计决策

 ┌───────────────────┬────────────────────────┬───────────────────────────────────┐
 │       问题        │          决策          │               原因                │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ STN输入旋转(训练) │ bilinear               │ nearest梯度为零，AngleCNN无法训练 │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ STN输入旋转(推理) │ nearest                │ 保持One-hot离散性                 │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ STN输出旋转       │ bilinear               │ logits连续值                      │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ AngleCNN输入      │ 原始图像               │ 避免循环依赖                      │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 旋转不变性        │ 显式等变性损失         │ 不能假设SAE自动学到               │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 解码器展开        │ FC→4×4→6步上采样       │ 避免FC(63,65536)过拟合            │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 解码器末端        │ 8ch→1×1Conv→3ch        │ 避免16→3突变                      │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 预测器归一化      │ LayerNorm              │ 对batch size不敏感                │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 预测器batch_size  │ 128                    │ MLP简单，大batch更稳              │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ λ_angle           │ 0.5                    │ 补偿63维vs2维量级差               │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 索引对齐          │ h5存indices            │ Phase2/3数据一致                  │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 角度表征          │ [cosθ,sinθ]            │ 解决0°/360°拓扑跳变               │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ L1正则            │ 只z_shape              │ θ物理量不稀疏化                   │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ Phase2编码        │ eval()+no_grad()       │ BN running stats一致              │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ 循环移位axis      │ axis=0(31电极)         │ 确认与存储格式对齐                │
 ├───────────────────┼────────────────────────┼───────────────────────────────────┤
 │ ReduceLROnPlateau │ patience=10,factor=0.5 │ 200epoch合理                      │
 └───────────────────┴────────────────────────┴───────────────────────────────────┘

 验证

 1. python scripts/train.py --method sae --max-iters 2 --hdf5-path dataset_sim/level_1/data.h5
 2. Phase1后：100张图×4旋转角度，z_shape MSE < 0.01 且角度准确跟随
 3. python scripts/train.py --method sae_predictor --max-iters 2 --hdf5-path dataset_sim/level_1/data.h5
 4. python scripts/evaluate_all.py --methods sae --levels 1
