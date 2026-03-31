  # ST-1D-VQ-EIT 新增独立方法实现方案                                                                                                      
                                                                                                                                           
  ## Summary                                                                                                                               
                                                                                                                                           
  在现有连续 sae / sae_predictor 体系之外，新增一套独立的离散字典链路：                                                                    
                                                                                                                                           
  - vq_sae：训练 ST-1D-VQ-VAE，从 GT 图像学习“标准姿态 + 1D 全局离散槽位”表示。                                                            
  - vq_sae_predictor：用纯 MLP 从 2356 维测量预测 N=16 个离散槽位分类和 2 维角度向量。                                                     
  - vq_sae_pipeline：推理时 测量 -> 槽位分类 argmax -> 冻结 VQ decoder -> 旋转回位 -> 分割图。                                             
  - 评估、可视化、latent 缓存、训练入口全部单独注册，不覆盖现有 sae 结果与脚本。                                                           

  默认实现按你确认的设置：                                                                                                                 
                                                                                                                                           
  - 独立新增方法，不替换旧 SAE                                                                                                             
  - N=16, K=512                                                                                                                            
  - predictor 训练时保留“角度头回归 + 冻结 VQ-AE 解码器”                                                                                   
                                                                                                                                           
  ## Key Changes                                                                                                                           
                                                                                                                                           
  ### 1. 新增 VQ-AE 模型与训练链路                                                                                                         
                                                                                                                                           
  新增一组独立模块：                                                                                                                       
                                                                                                                                           
  - src/models/vq_sae/model.py
  - src/models/vq_sae/predictor.py                                                                                                         
  - src/configs/vq_sae_config.py                                                                                                           
  - src/configs/vq_sae_predictor_config.py                                                                                                 
  - src/trainers/vq_sae_trainer.py                                                                                                         
  - src/trainers/vq_sae_predictor_trainer.py                                                                                               
  - src/pipelines/vq_sae_pipeline.py                                                                                                       
                                                                                                                                           
  vq_sae 模型结构按文档实现：                                                                                                              
                                                                                                                                           
  - AngleCNN：预测 angle_xy = [cosθ, sinθ]                                                                                                 
  - STN 逆旋转到标准姿态                                                                                                                   
  - EncoderCNN -> flatten -> FC -> (B, N, D) 连续槽位表示                                                                                  
  - VectorQuantizer1D：对每个槽位在 K=512 个 codebook 向量中做最近邻量化                                                                   
  - Decoder：量化嵌入拼接后经 FC -> 4x4 -> CNN 上采样 -> canonical logits                                                                  
  - STN 正向旋转回原角度，输出最终 logits                                                                                                  
                                                                                                                                           
  默认超参数：                                                                                                                             
                                                                                                                                           
  - num_slots = 16                                                                                                                         
  - codebook_size = 512                                                                                                                    
  - code_dim 作为 YAML 配置项，默认建议 32                                                                                                 
  - shape latent 不再是连续 63 维，而是 16 个离散索引                                                                                      
                                                                                                                                           
  VQ 损失使用标准组合：                                                                                                                    
                                                                                                                                           
  - recon_loss = CE + Dice                                                                                                                 
  - commitment_loss                                                                                                                        
  - codebook_loss 或 EMA codebook 更新二选一                                                                                               
    实现时固定一种，避免同仓库内再出现双实现分叉。                                                                                         
    推荐默认先用普通 straight-through VQ + beta commitment，便于调试和保存。                                                               
                                                                                                                                           
  ### 2. latent 缓存改为“离散索引 + 角度”                                                                                                  
                                                                                                                                           
  vq_sae 训练完成后自动执行编码缓存，生成新的 latent_codes.h5，内容改为：                                                                  
                                                                                                                                           
  - indices: (N_samples, 16)，int64                                                                                                        
  - angle_xy: (N_samples, 2)，float32                                                                                                      
  - sample_indices: (N_samples,)，与原 HDF5 数据索引对齐                                                                                   
  - 可选保存 codebook_size, num_slots, code_dim 作为 attrs                                                                                 
                                                                                                                                           
  不再保存连续 codes: (N,65) 这种旧 SAE 语义。                                                                                             
  旧 sae 的 latent 缓存格式保持不动，两个体系并存。                                                                                        
                                                                                                                                           
  对应新增/扩展数据集类：                                                                                                                  
                                                                                                                                           
  - VQGTHDF5Dataset：给 vq_sae 读 GT                                                                                                       
  - VQSAEPredictorHDF5Dataset：给 vq_sae_predictor 读 measurements + slot indices + angle_xy + gt_indices                                  
    保持和现有 HDF5 索引对齐逻辑一致。                                                                                                     
                                                                                                                                           
  ### 3. predictor 改成“多槽位分类 + 角度回归”                                                                                             
                                                                                                                                           
  vq_sae_predictor 的 MeasurementPredictor 改为：                                                                                          
                                                                                                                                           
  - MLP backbone：2356 -> 512 -> 256 -> 128                                                                                                
  - head_angle: 输出 (B, 2)，F.normalize                                                                                                   
  - head_slots: 输出 (B, 16, 512) logits                                                                                                   
                                                                                                                                           
  训练损失按文档固定为：                                                                                                                   
                                                                                                                                           
  - slot_loss = mean over slots of CrossEntropy(slot_logits[:, i, :], target_indices[:, i])                                                
  - angle_loss = MSE(pred_angle_xy, target_angle_xy)                                                                                       
  - total_loss = slot_loss + lambda_angle * angle_loss                                                                                     
    其中 lambda_angle 继续保留为 YAML 可配，默认 0.5                                                                                       
                                                                                                                                           
  训练时使用冻结的 vq_sae 解码器只做验证可视化和可选辅助前向，不回传 decoder 梯度。                                                        
  按你的选择，主训练目标仍是分类槽位 + 角度回归，不做 decoder 联合微调。                                                                   
                                                                                                                                           
  ### 4. 推理与评估链路                                                                                                                    
                                                                                                                                           
  新增 VQSAEPipeline：                                                                                                                     
                                                                                                                                           
  - 读取 vq_sae_predictor 和 vq_sae 结果目录                                                                                               
  - 输入测量先减 Uelref、再应用 vincl                                                                                                      
  - MLP 输出：                                                                                                                             
      - slot_logits -> argmax -> discrete indices                                                                                          
      - angle_xy                                                                                                                           
  - 使用冻结 vq_sae 的 VQ decoder 从离散索引重建 canonical logits                                                                          
  - 旋转回位并 argmax 得到最终分割图                                                                                                       
                                                                                                                                           
  evaluate_all.py 增加新方法名：                                                                                                           
                                                                                                                                           
  - vq_sae                                                                                                                                 
                                                                                                                                           
  并支持通过新的 YAML 配置读取：                                                                                                           
                                                                                                                                           
  - vq_sae_dir                                                                                                                             
  - vq_sae_predictor_dir                                                                                                                   
    为空时自动取最新 {num}，与现有 sae_pipeline.yaml 风格一致，但建议独立成：                                                              
  - scripts/vq_sae_pipeline.yaml                                                                                                           
                                                                                                                                           
  ### 5. 训练入口、结果目录与兼容策略                                                                                                      
                                                                                                                                           
  train.py 新增两种 method：                                                                                                               
                                                                                                                                           
  - vq_sae                                                                                                                                 
  - vq_sae_predictor                                                                                                                       
                                                                                                                                           
  结果目录遵守现有规范：                                                                                                                   
                                                                                                                                           
  - results/vq_sae_baseline_{num}                                                                                                          
  - results/vq_sae_predictor_baseline_{num}                                                                                                
                                                                                                                                           
  如果后续需要 decoder 微调或其他分支，再新增明确后缀，不复用旧 SAE 命名。                                                                 
                                                                                                                                           
  现有 sae / sae_predictor / sae_pipeline / latent_codes.h5 全部保留，不做破坏性修改。                                                     
  可视化脚本后续如需支持新链路，单独扩展，不混用旧 SAE 语义。                                                                              
                                                                                                                                           
  ## Test Plan                                                                                                                             
                                                                                                                                           
  ### 冒烟与形状验证                                                                                                                       
                                                                                                                                           
  - python scripts/train.py --method vq_sae --max-iters 2 --hdf5-path dataset_sim/level_1/data.h5                                          
      - 确认前向、VQ 量化、loss、checkpoint、latent cache 生成正常                                                                         
  - python scripts/train.py --method vq_sae_predictor --max-iters 2 --hdf5-path dataset_sim/level_1/data.h5 --vq-sae-checkpoint ...        
    --latent-h5-path ...                                                                                                                   
      - 确认分类目标、角度目标、冻结 decoder、batch 训练正常                                                                               
  - 检查 latent_codes.h5 字段和 dtype：                                                                                                    
      - indices 为整数                                                                                                                     
      - angle_xy 为 float32                                                                                                                
      - 样本索引与原 HDF5 对齐                                                                                                             
                                                                                                                                           
  ### 功能正确性                                                                                                                           
                                                                                                                                           
  - 用 vq_sae 对官方 GT 做自编码重建可视化：                                                                                               
      - 检查重建是否规则、低频、无破碎异形                                                                                                 
  - 用 vq_sae_predictor + vq decoder 在仿真 train/val/test 抽样可视化：                                                                    
      - 看预测是否回到与 GT 同域                                                                                                           
  - evaluate_all.py --methods vq_sae --levels 1                                                                                            
      - 跑通官方评估集推理                                                                                                                 
                                                                                                                                           
  - 随机旋转增强后：
      - angle_xy 同步旋转
      - slot indices 保持不变
      - 图像监督若有验证可视化时，旋转目标一致
  - predictor checkpoint 不应包含 decoder 可训练状态；pipeline 必须从 vq_sae checkpoint 读 decoder
  - 恢复训练时 best.pt / last.pt / training_log / scheduler / optimizer 均可继续使用

  ## Assumptions

  - 新方案作为独立方法落地，不替换现有 sae
  - 默认字典规格固定为 N=16, K=512
  - 默认使用 CE + Dice 作为 vq_sae 重建损失
  - predictor 训练默认只优化槽位分类和角度回归，不做 decoder 联合微调
  - VQ 实现默认采用标准 straight-through 量化；不在 v1 同时支持 EMA 与非 EMA 两套分支
  - Notebook/可视化支持不作为首批核心链路，优先先把训练、缓存、推理、评估主链打通
