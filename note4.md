  已完成

  新建 8 个文件：
  1. src/models/sae/__init__.py — 导出
  2. src/models/sae/model.py — ST-SAE (AngleCNN + EncoderCNN + Decoder + STN旋转)，6.86M 参数
  3. src/models/sae/predictor.py — 纯 MLP 双头预测器 (LayerNorm)，1.38M 参数
  4. src/configs/sae_config.py — SAE 训练配置
  5. src/configs/sae_predictor_config.py — 预测器训练配置
  6. src/trainers/sae_trainer.py — Phase 1 训练 + Phase 2 自动编码保存
  7. src/trainers/sae_predictor_trainer.py — Phase 3 预测器训练（含旋转增强）
  8. src/pipelines/sae_pipeline.py — 推理管线（含软阈值截断）

  修改 7 个文件：
  - src/data/sim_dataset.py — +GTHDF5Dataset, +SAEPredictorHDF5Dataset
  - src/data/__init__.py, src/configs/__init__.py, src/trainers/__init__.py, src/pipelines/__init__.py — 注册
  - scripts/train.py — +sae, +sae_predictor 方法 + --sae-checkpoint/--latent-h5-path 参数
  - scripts/evaluate_all.py — +sae 评估

  训练命令：
# Phase 1+2: SAE 训练
python scripts/train.py --method sae --hdf5-path dataset_sim/level_1/data.h5

# Phase 3: 预测器训练
python scripts/train.py --method sae_predictor --hdf5-path dataset_sim/level_1/data.h5 \
--sae-checkpoint results/sae_baseline_1/best.pt \
--latent-h5-path results/sae_baseline_1/latent_codes.h5


• 当前你这套 SAE 的学习率调度策略是 ReduceLROnPlateau，不是 warmup/cosine。
              
  Phase 1: sae
  定义在 src/trainers/sae_trainer.py 和 src/configs/sae_config.py：           
              
  - 优化器：AdamW   
  - 初始学习率：1e-3
  - weight_decay = 1e-4
  - 调度器：ReduceLROnPlateau(mode='min')
  - scheduler_patience = 10
  - scheduler_factor = 0.5
  - min_lr = 1e-5

  也就是：

  - 如果监控指标连续 10 个 epoch 没下降
  - 学习率乘 0.5
  - 最低不低于 1e-5

  Phase 3: sae_predictor
  定义在 src/trainers/sae_predictor_trainer.py 和 src/configs/sae_predictor_config.py：

  - 优化器：AdamW
  - 初始学习率：3e-4
  - weight_decay = 1e-4
  - 调度器：ReduceLROnPlateau(mode='min')
  - scheduler_patience = 10
  - scheduler_factor = 0.5

  这里配置里没有单独设置 min_lr，所以是否有下限取决于 ReduceLROnPlateau 默认行为；当前 trainer 代码里也没有显式传 min_lr。

  监控哪个指标
  这部分由 src/trainers/base_trainer.py 控制：

  - 优先用 val_loss
  - 如果没有 val_loss，回退到训练 avg_loss

  你这次是 HDF5 自动划分了 train/val/test，所以当前 sae 训练时：

  - scheduler 实际上会看 val_loss                
              
  早停        
  顺带一起说，当前配置是：                       
              
  - sae: early_stopping_patience = 30            
  - sae_predictor: early_stopping_patience = 30  
              
  也就是连续 30 个 epoch 没改善就早停。          
              
  如果你愿意，我下一步可以继续帮你判断：         
  这套 patience=10, factor=0.5 对 SAE 是否偏保守，需不需要调快一点。          


best val_loss: 0.00784


  1. 普通 sae 训练                          
  现在在 src/trainers/sae_trainer.py 里，流程变成：                      
         
  - gt_onehot -> encode -> z_shape, angle_xy
  - 对 z_shape 加微弱高斯噪声               
  - decode(z_shape_noisy, angle_xy)         
  - 再计算重建损失                          
         
  也就是说噪声只加在：                      
         
  - z_shape（63 维形态向量）                
         
  不会加在：   
         
  - angle_xy（2 维角度）                    
         
  这和你的设计目标一致。                    
         
  默认噪声强度我设置为：                    
         
  - latent_noise_std = 0.05                 
         
  位置在：     
         
  - src/configs/sae_config.py               
         
  这个量级是按你前面 sae_predictor 的误差水平来定的：                    
         
  - 比你当前 predictor 的整体 RMSE 量级略小 
  - 适合作为 decoder 鲁棒性训练噪声         
         
  2. 已有 SAE 权重的 decoder-only fine-tune 
  现在支持：   
         
  - 加载训练好的 SAE checkpoint             
  - 冻结：     
      - angle_cnn                           
      - encoder
  - 只训练：   
      - decoder
         
  训练流程是： 
         
  - 先用冻结编码器得到 z_shape, angle_xy    
  - 再对 z_shape 加噪声                     
  - 用 noisy latent 训练 decoder 重建       
         
  对应实现也在：                            
         
  - src/trainers/sae_trainer.py             
         
  3. 新增命令行参数                         
  在 scripts/train.py 里新增了：            
         
  - --latent-noise-std                      
  - --decoder-finetune                      
         
  并且复用了原来的：                        
         
  - --sae-checkpoint                        
         
  所以你现在可以这样用。                    
         
普通 SAE 训练，默认噪声：                 
        
python scripts/train.py --method sae --hdf5-path dataset_sim/level_1/data.h5 
        
显式指定 latent 噪声：                    
        
python scripts/train.py --method sae --hdf5-path dataset_sim/level_1/data.h5 --latent-noise-std 0.05      
        
加载已有 SAE，冻结编码器，只训练 decoder：
        
python scripts/train.py --method sae --hdf5-path dataset_sim/level_1/data.h5 --decoder-finetune --sae-checkpoint results/              
sae_baseline_3/best.pt                    
        
如果你想把噪声调得更保守一点，也可以：    
        
python scripts/train.py --method sae --hdf5-path dataset_sim/level_1/data.h5 --decoder-finetune --sae-checkpoint results/              
sae_baseline_3/best.pt --latent-noise-std 0.03                         
         
  4. 当前设计上的取舍                       
  我现在这样处理的是：                      
         
  - 训练时：z_shape 加噪声                  
  - 验证时：不加噪声                        
  - 所以 val_loss 仍然反映 clean latent 下的重建质量                     
         
  这比较稳，也方便你观察：
         
  - decoder 鲁棒性训练是否损害了 clean reconstruction                    
         
  如果你愿意，我下一步可以继续帮你加一个：  
         
  - --val-latent-noise-std                  
    让验证也能评 noisy latent 下的表现。    



• sae_predictor 的损失在 src/trainers/sae_predictor_trainer.py 里定义得很直接：                
  
  pred_shape, pred_angle = self.model(measurements)               
  
  mse_shape = F.mse_loss(pred_shape, target_shape)                
  mse_angle = F.mse_loss(pred_angle, target_angle)                
  lambda_angle = self.config.training.lambda_angle                
  total_loss = mse_shape + lambda_angle * mse_angle               

  也就是：                     
  
  - target_z 前 63 维是 z_shape
  - 后 2 维是 angle_xy = [cosθ, sinθ]
  - 总损失是                   
    MSE(z_shape) + 0.5 * MSE(angle_xy)                            
  
  其中 lambda_angle=0.5 来自 src/configs/sae_predictor_config.py。
  
  低维向量的数值范围           
  
  MeasurementPredictor 在 src/models/sae/predictor.py 里是：      
  
  - head_shape: 线性层直接输出 z_shape                            
  - head_angle: 线性层输出后再 F.normalize(...)                   
  
  所以：                       
  
  - z_shape：                  
      - 没有硬范围约束         
      - 是实数，理论上可取任意值     
      - 但因为 SAE 训练时加了 L1 稀疏约束，实际通常会偏小、偏稀疏 
  - angle_xy：                 
      - 经过 L2 normalize      
      - 满足 cos^2 + sin^2 = 1 
      - 每个分量都在 [-1, 1]   
  
  best val_loss = 0.00784 代表什么   
  
  这个值是：                   
  
  - mse_shape + 0.5 * mse_angle
  
  不是单独某一项。             
  
  如果粗略把它当成“平均平方误差”的量级来看：                      
  
  - sqrt(0.00784) ≈ 0.0885     
  
  也就是一个比较粗的 RMSE 量级大约在 0.09 左右。                  
  
  这个误差大不大？             
  
  从当前这套任务看，不算大，属于比较好的水平，原因有几个：        
  
  1. angle_xy 是单位圆上的 2 维向量  
     这一项如果预测得很差，损失不会这么低。                       
  2. z_shape 是无界实数        
     在这种情况下，总 val_loss 能压到 1e-2 以下，一般说明 latent 回归已经比较准了。            
  3. 你这个 predictor 最终还要接 SAE decoder 重建图像             
     真正更关键的是：          
      - 用 predictor 预测出的 latent 解码后，图像质量和 score 怎么样                           
      - 而不是只看 latent MSE 本身   
  
  所以更实用的判断是：         
  
  - 0.00784 从 latent 回归角度看是不错的                          
  - 不属于“大误差”             
  - 但最终还要看：             
      - scripts/evaluate_all.py --methods sae                     
      - 或解码后的可视化效果   
  
  如果你愿意，我可以继续帮你把这个 0.00784 拆成：                 
  
  - mse_shape                  
  - mse_angle                  
    分别看哪一项占主导。       
