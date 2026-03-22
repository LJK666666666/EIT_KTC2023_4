● 整个 src/ 和 scripts/ 都是新增的（未跟踪），因为是在第一次 commit 之后创建的。让我总结一下训练框架的完整实现结果。
                                                                                                                                                                       ---                                                                                                                                                                  训练代码框架已全部实现并验证通过。以下是完整的实现清单：
                                                                                                                                                                     
  新增文件

  核心训练框架：
  - src/trainers/__init__.py — 包导出
  - src/trainers/base_trainer.py — BaseTrainer ABC，提供自动递增结果目录、checkpoint 保存/加载/恢复、TensorBoard 日志、training_log.json、max_iters 快速测试
  - src/trainers/fcunet_trainer.py — FCUNet 两阶段训练（15 epoch init_linear MSE + 500 epoch 全模型 CrossEntropy），随机 level 增强
  - src/trainers/postp_trainer.py — PostP 单阶段训练（ConcatDataset 7 个 level，CrossEntropy）
  - src/trainers/condd_trainer.py — CondD 扩散模型逐 level 训练（epsilon/score loss，EMA warm-start，DDIM 验证采样）

  数据相关（之前会话已创建）：
  - src/data/phantom_generator.py — 随机 phantom 图像生成
  - src/data/sim_dataset.py — FCUNetTrainingData / SimData / MmapDataset

  配置增强（之前会话已创建）：
  - src/configs/base_config.py — get_base_training_config() 共享训练参数
  - src/configs/fcunet_config.py — 增加 init_epochs, lr, scheduler, 数据路径等
  - src/configs/postp_config.py — 增加 level_to_num, mmap, scheduler
  - src/configs/condd_config.py — 增加 ema_decay, data paths, level_to_num

  入口脚本：
  - scripts/train.py — 统一训练入口，支持 --method {fcunet,postp,condd}, --max-iters, --resume, --batch-size 等
  - scripts/generate_data.py — 训练数据生成，支持 --level, --all-levels, --save-measurements

  验证结果

  - 所有模块导入 ✓
  - 3 个配置正确加载 ✓
  - 3 个模型构建 + 参数量确认 ✓
  - Checkpoint 保存/加载/恢复 ✓（包括 EMA 和两阶段状态）
  - 结果目录自动递增 ✓
  - 前向+反向传播全部通过 ✓

  使用方式

# 训练 FCUNet（需要先生成训练数据）
python scripts/train.py --method fcunet

# 训练 PostP
python scripts/train.py --method postp

# 训练 CondD level 3
python scripts/train.py --method condd --level 3

# 快速测试（1 iteration）
python scripts/train.py --method postp --max-iters 1 --batch-size 2

# 恢复训练
python scripts/train.py --method fcunet --resume results/fcunet_baseline_1/last.pt

# 生成训练数据
python scripts/generate_data.py --level 3 --num-images 2000

24.2s/sample avg

  训练前需要按顺序执行以下命令：

  第1步：生成验证用的 5 通道重建（PostP/CondD 验证需要）
python scripts/generate_val_reco.py
  这会对 4 张挑战图像 × 7 个难度级别生成 ChallengeReconstructions/level_{1-7}/reco{1-4}.npy。

  第2步：生成训练数据（每个 level 至少需要几个样本来测试）
# 快速测试：每个 level 生成 3 个样本
python scripts/generate_data.py --all-levels --num-images 3

# 正式训练时需要更多（如 2000+/level），耗时较长

第3步：快速测试训练
# PostP 快速测试（1 次迭代）
python scripts/train.py --method postp --max-iters 1

# CondD 快速测试
python scripts/train.py --method condd --level 1 --max-iters 1

# FCUNet 快速测试（需要 dataset/gt/ 和 dataset/measurements/ 目录）
python scripts/train.py --method fcunet --max-iters 1

  注意：FCUNet 的训练数据格式不同（需要 dataset/gt/ + dataset/measurements/），需要运行：
  python scripts/generate_data.py --level 1 --num-images 3 --save-measurements
  并将 gt 和 measurements 目录从 dataset/level_1/ 移到 dataset/ 根目录下（或修改 fcunet_config.py 中的 dataset_base_path）。



  核心逻辑对比                                                                                                                                    
                                                                                                                                                    ┌──────────────┬─────────────────────────────────────────────────────┬──────────────────────────────────────────────┬───────────────────────┐   
  │     步骤     │           原版 (create_training_data.py)            │          我们的 (generate_data.py)           │        一致？         │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤
  │ 加载参考数据 │ loadmat('TrainingData/ref.mat') → Injref, Mpat      │ loadmat(ref_path) → Injref, Mpat             │ 一致 (路径参数化)     │
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 加载网格     │ load_mesh("Mesh_dense.mat")                         │ load_mesh(mesh_name)                         │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 正演求解器   │ EITFEM(mesh2, Injref, Mpat, vincl)                  │ 同                                           │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 噪声参数     │ noise_std1=0.05, noise_std2=0.01                    │ 同                                           │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 参考测量模拟 │ sigma_bg=0.745, SolveForward + InvLn噪声            │ 同                                           │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ vincl掩码    │ 手写循环(L89-97)                                    │ create_vincl() 封装了同样的循环              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 重建器       │ LinearisedRecoFenics(Uelref,B,vincl_level,          │ 同，base_path='KTC2023_SubmissionFiles/data' │ 一致                  │   
  │              │ mesh_name="sparse") base_path默认"data"             │                                              │ (路径适配项目根目录)  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ Phantom生成  │ create_phantoms()                                   │ 同                                           │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 电导率采样   │ bg=0.745, resistive=rand*0.1+0.025,                 │ 同                                           │ 一致                  │   
  │              │ conductive=rand+5.0                                 │                                              │                       │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 正演+加噪    │ SolveForward(sigma_gt, z) + InvLn*randn             │ 同                                           │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 5通道重建    │ reconstruct_list(Uel_noisy, alphas) →               │ 同                                           │ 一致                  │   
  │              │ 5个interpolate_to_image → np.stack                  │                                              │                       │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ alpha参数    │ level_to_alphas 字典                                │ LEVEL_TO_ALPHAS（数值完全相同）              │ 一致                  │   
  └──────────────┴─────────────────────────────────────────────────────┴──────────────────────────────────────────────┴───────────────────────┘   

  差异

  1. 缺少 HanddrawnImages 支持：原版有 use_handdrawn_images=True 选项，从作者本地路径 /home/adenker/projects/.../KTC_handdrawn_images
  加载手绘图像作为额外的 phantom 来源。我们只用了 create_phantoms() 随机生成。这个手绘数据集不在提交文件中，无法直接复现。
  2. 缺少 conductivity values pickle：原版额外保存了 sigmavalues_ztm_{idx}.pkl（包含 background/resistive/conductive
  三个电导率值），我们没保存。这个文件训练时不使用，仅供调试。
  3. 文件命名起始偏移：原版 max_image_idx=2000，我们默认 start_idx=0。不影响训练（数据集按目录遍历文件名）。

  总结：核心数据生成逻辑完全一致。 差异都是外围功能（手绘图像数据源不可获取、调试用pickle文件、文件名偏移），不影响训练数据的正确性。



  修改了 3 个文件 + 新建 1 个文件：                                                                                                                
                                                                                                                                                     ┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐    │                 文件                  │                                                改动                                                 │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ src/ktc_methods/KTCFwd.py             │ EITFEM 加 use_gpu 参数，SolveForward 中 spsolve → CuPy 密集求解                                     │
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/reconstruction/linearised_reco.py │ 加 use_gpu; reconstruct_list 对角优化（CPU 也受益）+ CuPy GPU 路径; interpolate_to_image 缓存       │  
  │                                       │ Delaunay 三角化                                                                                     │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ scripts/generate_data.py              │ 加 --gpu 参数; 逐步计时 (phantom/forward/noise/reco/interp/io); 保存 timing JSON                    │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ scripts/benchmark_data_gen.py         │ 新建: CPU vs GPU 基准对比，输出对比表 + JSON                                                        │  
  └───────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘  

  三个层面的优化：

  1. SolveForward: scipy.sparse.linalg.spsolve → cupy.linalg.solve（密集，避免 CuPy spsolve 不支持矩阵 RHS 的问题）
  2. reconstruct_list: 消除了完整 M×M 对角矩阵创建（np.diag → 向量行缩放），CPU 和 GPU 都受益；GPU 路径全程 CuPy
  3. interpolate_to_image: Delaunay 三角化只算一次，缓存复用

  Colab 上的用法：

# 基准对比（保存到 results/gpu_benchmark.json）
python scripts/benchmark_data_gen.py --num-samples 10
python scripts/benchmark_data_gen.py --num-samples 10 --measurements-only

# GPU 加速生成数据
python scripts/generate_data.py --level 1 --num-images 1000 --measurements-only --gpu
python scripts/generate_data.py --level 1 --num-images 1000 --gpu


  已完成的 6 项优化：
  1. A0 刚度矩阵向量化组装 + COO（~10-15x）                                                                                                          2. S0 电极边界 COO 替代 CSR（消除 SparseEfficiencyWarning）
  3. reconstruct_list 对角矩阵优化（~2-3x）                                                                                                        
  4. reconstruct_list CuPy GPU 加速（6.8x）
  5. Delaunay 三角化缓存（~1.5-2x）
  6. SolveForward 回退 CPU 稀疏求解（修正性能倒挂）

  未来优化思路（按收益排序）：
  - 高收益：Numba JIT、S0 缓存、CPU 多进程流水线、HDF5 存储
  - 中等收益：稀疏 Cholesky、CuPy 稀疏求解、积分常量预计算
  - 大规模专用：S0 全向量化、混合精度 float32、CUDA 自定义核函数


  每个 worker 初始化需要 8.6s（其中 Python 模块导入 7.2s），但 40 样本只有 10 样本/worker × 356ms = 3.6s 的实际计算。初始化是计算量的 2.4 倍。 

  再加上 4 个 worker 的初始化阶段竞争磁盘 I/O 和 CPU，实际并行度很低 → 几乎串行执行。

  结论：多进程只在大样本量时有效：

  ┌────────┬────────────┬────────────┬────────┐
  │ 样本数 │  1 worker  │ 4 workers  │ 加速比 │
  ├────────┼────────────┼────────────┼────────┤
  │ 40     │ 572ms/样本 │ 305ms/样本 │ 1.9x   │
  ├────────┼────────────┼────────────┼────────┤
  │ 200    │ 399ms/样本 │ 132ms/样本 │ 3.0x   │
  ├────────┼────────────┼────────────┼────────┤
  │ 1000   │ 365ms/样本 │ 98ms/样本  │ 3.7x   │
  ├────────┼────────────┼────────────┼────────┤
  │ 5000   │ 358ms/样本 │ 91ms/样本  │ 3.9x   │
  └────────┴────────────┴────────────┴────────┘

  实际数据生成（2000+ 样本）时，4 workers 可达 ~3.7x 吞吐提升。40 样本的 benchmark 不适合测量多进程效果——初始化开销主导。


# 仅分析（默认 val_loss + patience=15）
python scripts/analyze_early_stopping.py

# 指定文件夹
python scripts/analyze_early_stopping.py results/fcunet_scaling_n100_4

# 分析 + 截断（自动备份 .bak）
python scripts/analyze_early_stopping.py --truncate results/fcunet_scaling_n100_4

# 自定义 patience
python scripts/analyze_early_stopping.py --patience 10 --truncate results/fcunet_scaling_n100_4



• 这里的 SSIM 实现就在 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py。
                                                                                                                                              
  evaluate_all.py 用的是 官方版 scoring_function()，不是训练时的快速版。流程是：                                                              
                                                                                                                                              
  1. 先把 3 类分割图拆成两张二值图                                                                                                            
     在 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:189 里：                                                               
                                                                                                                                              
  - class 2 一张 mask                                                                                                                         
  - class 1 一张 mask                                                                                                                         
                                                                                                                                              
  也就是：                                                                                                                                    
                                                                                                                                              
  - truth_c / reco_c：值等于 2 的位置记为 1，其余为 0                                                                                         
  - truth_d / reco_d：值等于 1 的位置记为 1，其余为 0                                                                                         
                                                                                                                                              
  2. 分别对这两张二值图调用 _ssim_official()                                                                                                  
     位置在 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:140                                                                
                                                                                                                                              
  这个函数里具体做了：                                                                                                                        
                                                                                                                                              
  - 常数：                                                                                                                                    
      - c1 = 1e-4                                                                                                                             
      - c2 = 9e-4                                                                                                                             
      - r = 80                                                                                                                                
  - 构造一个很大的二维高斯核                                                                                                                  
    见 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:154                                                                     
      - 窗口半径参数 r=80                                                                                                                     
      - 实际卷积核边长是 321 x 321                                                                                                            
  - 用 scipy.signal.convolve2d 计算局部均值                                                                                                   
      - gt：truth 的局部均值                                                                                                                  
      - gr：reco 的局部均值                                                                                                                   
  - 再计算局部方差和协方差                                                                                                                    
      - sigma_t2                                                                                                                              
      - sigma_r2                                                                                                                              
      - sigma_tr                                                                                                                              
  - 然后按 SSIM 标准公式逐像素得到 ssimimage：                                                                                                
    [                                                                                                                                         
    \text{SSIM}=\frac{(2\mu_t\mu_r+c_1)(2\sigma_{tr}+c_2)}                                                                                    
    {(\mu_t^2+\mu_r^2+c_1)(\sigma_t^2+\sigma_r^2+c_2)}                                                                                        
    ]                                                                                                                                         
                                                                                                                                              
  对应代码在：                                                                                                                                
                                                                                                                                              
  - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:181                                                                        
  - 最后对整张 ssimimage 取平均，得到这一类的分数                                                                                             
    见 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:185                                                                     
                                                                                                                                              
  3. 两类再取平均                                                                                                                             
     在 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:217 和 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/            
     scoring.py:224 分别算出：                                                                                                                
                                                                                                                                              
  - score_c                                                                                                                                   
  - score_d                                                                                                                                   
                                                                                                                                              
  最后：                                                                                                                                      
                                                                                                                                              
  - score = 0.5 * (score_c + score_d)                                                                                                         
    见 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:226                                                                     
                                                                                                                                              
  所以这里的 SSIM 不是直接对原始 0/1/2 标签图整体计算，而是：                                                                                 
                                                                                                                                              
  - 先转成两张二值 mask                                                                                                                       
  - 各算一次官方 SSIM                                                                                                                         
  - 最后平均                                                                                                                                  
                                                                                                                                              
  另外还有一个 FastScoringFunction()，位置在 /D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py:315：                             
                                                                                                                                              
  - 指标定义一样                                                                                                                              
  - 只是把二维卷积换成了可分离一维卷积近似加速                                                                                                
  - 训练时常用它                                                                                                                              
  - 正式评估时 evaluate_all.py 用的还是 scoring_function()                                                                                    

• CuPy 的 cupyx.scipy.sparse.linalg 在这台机器上缺 cublasLt*.dll，现成的 GPU CG 路线跑不起来。我不打算卡死在环境问题上，下一步改用这台机器已经可 
  用的 PyTorch CUDA 稀疏 CSR 来实现同样的 Block PCG，这样还能继续保留 GPU 加速尝试。

● KTC2023 提供了 32 电极的完整位置信息，定义在网格文件和代码中：

  电极布局

  - 数量：32 个电极，均匀分布在圆形容器边界上
  - 容器直径：230mm（半径 115mm）
  - 角间距：360° / 32 = 11.25°
  - 电极宽度：5.625°（占半个间距）
  - 间隙宽度：5.625°（电极与间隙等宽）

  各电极中心角度

  第 1 个电极中心在 2.8125°，之后每隔 11.25° 一个：
  2.8125°, 14.0625°, 25.3125°, 36.5625°, ... (共 32 个)

  计算公式（src/ktc_methods/KTCMeshing.py 中 create2Dmesh_circ）：
  elwidth = 360 / (2 * Nel)  # 5.625°
  elcenterangles = np.arange(elwidth/2, 360, 2*elwidth)

  数据来源

  ┌──────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │               文件               │                           内容                           │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Mesh_dense.mat / Mesh_sparse.mat │ 电极边界节点索引（elfaces, elfaces2）、网格拓扑          │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ ref.mat                          │ Injref(32×76 注入模式)、Mpat(测量模式)、Uelref(参考电压) │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ src/ktc_methods/KTCMeshing.py    │ 圆形网格生成，定义电极角度和宽度                         │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ src/ktc_methods/KTCFwd.py        │ 电极接触阻抗 z = 1e-6 Ω                                  │
  └──────────────────────────────────┴──────────────────────────────────────────────────────────┘

  测量条件

  - 注入模式：76 种电流注入模式（相邻电极对注入）
  - 接触阻抗：1×10⁻⁶ Ω（近乎理想接触）
  - 难度级别：Level 1-7 逐步移除前 2/4/.../12 个电极的数据
  - 重建网格：256×256 像素，覆盖 230mm×230mm 区域


● 拟合结果汇总

  电导率估计

  ┌────────────┬─────────────┬─────────────────┬──────────────────┐
  │    样本    │ σ_bg (背景) │ σ_1 (resistive) │ σ_2 (conductive) │
  ├────────────┼─────────────┼─────────────────┼──────────────────┤
  │ Ref (均匀) │ 0.804       │ —               │ —                │
  ├────────────┼─────────────┼─────────────────┼──────────────────┤
  │ Sample 1   │ 0.803       │ ≈0 (触底)       │ 9.70             │
  ├────────────┼─────────────┼─────────────────┼──────────────────┤
  │ Sample 2   │ 0.796       │ ≈0 (触底)       │ 4.00             │
  ├────────────┼─────────────┼─────────────────┼──────────────────┤
  │ Sample 3   │ 0.818       │ —               │ 4.37             │
  ├────────────┼─────────────┼─────────────────┼──────────────────┤
  │ Sample 4   │ 0.863       │ ≈0 (触底)       │ —                │
  └────────────┴─────────────┴─────────────────┴──────────────────┘

  - 背景电导率：~0.80-0.86 S/m（你的仿真用的 0.745 偏低）
  - Resistive 包裹体：电导率极低，趋近 0（几乎绝缘），你仿真中的 rand()*0.1+0.025 范围合理
  - Conductive 包裹体：4.0-9.7 S/m，你仿真中的 rand()+5.0 覆盖了中间段但上限偏低

  噪声水平

  ┌────────────────────────┬──────────┐
  │          指标          │    值    │
  ├────────────────────────┼──────────┤
  │ 残差 std               │ ~0.035   │
  ├────────────────────────┼──────────┤
  │ SNR                    │ ~21.5 dB │
  ├────────────────────────┼──────────┤
  │ 相对拟合误差           │ ~8.4%    │
  ├────────────────────────┼──────────┤
  │ 通道间方差比 (max/min) │ ~84000x  │
  ├────────────────────────┼──────────┤
  │ 噪声/信号比 (median)   │ ~4.0%    │
  └────────────────────────┴──────────┘

  残差 8.4% 比较大，可能还包含：2D 模型截断误差（真实水箱是
  3D）、电极位置偏差、网格离散化误差等。纯硬件噪声应该小于这个值。通道间方差比极大（84000x），说明噪声不是白噪声，有很强的空间结构性。


  scripts/generate_data.py 修改

  ┌─────────────────┬──────────────────────┬───────────────────────────────────┐
  │      项目       │         原值         │               新值                │
  ├─────────────────┼──────────────────────┼───────────────────────────────────┤
  │ 背景电导率      │ 0.745                │ 0.804                             │
  ├─────────────────┼──────────────────────┼───────────────────────────────────┤
  │ Conductive 范围 │ rand()+5.0 (5.0-6.0) │ *rand()2.0+4.0 (4.0-6.0)          │
  ├─────────────────┼──────────────────────┼───────────────────────────────────┤
  │ 系统偏差        │ 无                   │ 自动加载 data/systematic_bias.npy │
  └─────────────────┴──────────────────────┴───────────────────────────────────┘

  系统偏差用法

  # 默认：自动检测 data/systematic_bias.npy，存在则加载
  python scripts/generate_data.py --level 1 --num-images 100 --save-measurements

  # 指定路径
  python scripts/generate_data.py --level 1 --num-images 100 --sys-bias path/to/bias.npy

  # 禁用（指定不存在的路径）
  python scripts/generate_data.py --level 1 --num-images 100 --sys-bias none

  系统偏差向量（2356 维）会加到每个样本的带噪测量值上，模拟真实水箱中 2D/3D 截断误差、电极偏差等固定系统误差。


  src/data/advanced_phantom.py — 重写了形状生成逻辑：                                                                                                                     
                                                                                                                                                                          
  - 移除了布尔交并差（union/intersection/difference）操作                                                                                                                 
  - 新增 5 种轮廓风格，每个物体独立生成、互不重叠：                                                                                                                       
    - smooth：椭圆/圆弧（光滑边界）
    - wavy：正弦波径向扰动（有机轮廓，类似水滴/云朵）
    - angular：低顶点多边形 + 高尖锐度（锐角棱角）
    - star：交替内外半径（星形/凹边）
    - perlin：多谐波傅里叶调制半径（自然斑块）
  - create_phantoms_advanced()：放置 1-5 个不重叠物体，碰撞检测基于包围圆 + 像素级验证
  - create_phantoms_v2()：混合流水线（60% 轮廓 + 20% Perlin 阈值 + 20% Legacy），30% 概率叠加弹性形变

  notebooks/test_advanced_phantom.ipynb — 更新为 8 个可视化部分，新增了每种轮廓风格的独立展示画廊。

❯ 能不能改成用Keras3或JAX实现，方便GPU/TPU切换。

● 这个建议我觉得不太合理，原因如下：

  1. 整个代码库都是 PyTorch：BaseTrainer、DataLoader、checkpoint 机制、所有 pipeline、其他 3 个模型（FCUNet/PostP/CondD）全部是 PyTorch。仅 DPCA-UNet 用 Keras/JAX
  会导致两套训练框架并存，维护成本很高。
  2. 现有基础设施无法复用：BaseTrainer 的训练循环、ReduceLROnPlateau、早停、TensorBoard logging、checkpoint save/load 全部是 PyTorch API。改 Keras 需要重写整个训练器基类。  
  3. 数据管线不兼容：FCUNetTrainingData 返回的是 PyTorch Tensor，DataLoader 也是 PyTorch 的。Keras/JAX 需要完全不同的数据管线。
  4. PyTorch 本身已支持 GPU/TPU：torch_xla 可以在 TPU 上运行 PyTorch 代码，只需少量修改（设备改为 xla），不需要换框架。

  如果你的目标是在 Colab TPU 上训练，更务实的方案是：
  - 安装 torch_xla，把 device 从 cuda 改为 xla 即可
  - 不需要重写任何模型代码

  你确定要换 Keras/JAX 吗？还是说实际需求是 TPU 支持？

✻ Churned for 1m 31s

❯ 既然使用PyTorch为什么需要TensorFlow？

● 不需要。TensorFlow 不是代码的依赖，是你环境里恰好装了 TensorFlow，而 torch.utils.tensorboard 导入时会尝试加载它，结果因为 TensorFlow 和 NumPy 2.x 不兼容而崩溃。

  最简单的解决方案——卸载不需要的 TensorFlow：

  pip uninstall tensorflow -y

  或者如果你还需要 TensorFlow 做别的事，升级到兼容 NumPy 2 的版本：

  pip install tensorflow --upgrade

  使用方式：
  # Colab TPU
  pip install torch_xla
  python scripts/train.py --method dpcaunet --device tpu

  # 本地 GPU 不变
  python scripts/train.py --method dpcaunet --device cuda


Params: 4,130,371

python -c "
from src.models.fcunet import FCUNet
from src.configs import get_fcunet_config
c = get_fcunet_config()
m = FCUNet(image_size=256, in_channels=c.model.in_channels, model_channels=c.model.model_channels, out_channels=c.model.out_channels,
num_res_blocks=c.model.num_res_blocks, attention_resolutions=c.model.attention_resolutions, channel_mult=c.model.channel_mult, conv_resample=c.model.conv_resample,      
dims=c.model.dims, num_heads=c.model.num_heads, num_head_channels=c.model.num_head_channels, num_heads_upsample=c.model.num_heads_upsample,
use_scale_shift_norm=c.model.use_scale_shift_norm, resblock_updown=c.model.resblock_updown, use_new_attention_order=c.model.use_new_attention_order,
max_period=c.model.max_period)
print(sum(p.numel() for p in m.parameters()))
"
40,701,699

● 因为你没有加 --all-levels 参数，所以只生成了默认的 level 1。
                              
  # 生成全部 7 个 level
  python scripts/generate_data.py --all-levels --num-images 200 --workers 4 --start-idx 0 --measurements-only                                                                   
  # 或者只指定某个 level                                                                                                                                                     
  python scripts/generate_data.py --level 3 --num-images 200 --workers 4 --start-idx 0 --measurements-only                                                                   
  
  不过对于 FCUNet 和 DPCA-UNet，其实只需要 level 1 的数据就够了——训练时通过 vincl 掩码随机模拟不同 level 的电极缺失。--all-levels 主要是给 PostP 和 CondD
  用的，因为它们需要每个 level 单独的线性化重建结果作为输入。

python -c "
import os, yaml
from scripts.generate_data import _update_dataset_info

# Create a test h5
import h5py, numpy as np, tempfile
tmp = tempfile.mkdtemp()
h5_path = os.path.join(tmp, 'data.h5')
with h5py.File(h5_path, 'w') as f:
    f.create_dataset('gt', data=np.zeros((150, 256, 256), dtype=np.float32))
    f.create_dataset('measurements', data=np.zeros((150, 2356), dtype=np.float32))

_update_dataset_info(tmp, h5_path)

# Read back
info_path = os.path.join(tmp, 'dataset_info.yaml')
with open(info_path) as f:
    info = yaml.safe_load(f)
print(yaml.dump(info, default_flow_style=False, sort_keys=False))

# Clean up
import shutil
shutil.rmtree(tmp)
"


python scripts/generate_data.py --num-images 800 --workers 2 --start-idx 0 --measurements-only
Systematic bias loaded: data/systematic_bias.npy (mean=-1.7793e-04, std=3.8047e-02)
Using 2 workers for level 1 (400 samples/worker, chunk_size=1000)...
                                                                                                                                                 Level 1: 800 samples in 67.9s (85 ms/sample throughput)
Merging to HDF5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  8.67it/s]
  Warning: could not remove dataset\level_1\_batches, delete manually.
  gt: (800, 256, 256) (float32)
  measurements: (800, 2356) (float64)
  HDF5 saved to: dataset\level_1\data.h5
  dataset_info.yaml updated: 800 samples
Data generation complete.


  ┌───────────────┬────────────┬──────────────────┐
  │     方案      │ 每样本大小 │   读取代码改动   │
  ├───────────────┼────────────┼──────────────────┤
  │ float32（旧） │ 256 KB     │ —                │
  ├───────────────┼────────────┼──────────────────┤
  │ uint8 无压缩  │ 64 KB      │ —                │
  ├───────────────┼────────────┼──────────────────┤
  │ 2-bit 打包    │ 16 KB      │ 需要 pack/unpack │
  ├───────────────┼────────────┼──────────────────┤
  │ uint8 + gzip  │ 1.2 KB     │ 无需改动         │
  └───────────────┴────────────┴──────────────────┘

     Method         Size      Write  Rand Read   Seq Read
     ..........     (KB)   (ms/smp)   (ms/smp)   (ms/smp)
     ----------------------------------------------------
     raw          32032      0.040      0.017      0.015
     gzip-1         637      0.089      0.045          -
     gzip-4         475      0.208      0.103      0.097
     gzip-9         277      2.199      0.078          -
     lzf           1020      0.044      0.064      0.064



● 当前模型 4.1M 参数。可以调的旋钮和效果：

  ┌──────────────────┬──────────────┬────────────────────────────────────────────────────┐
  │       参数       │    当前值    │                        影响                        │
  ├──────────────────┼──────────────┼────────────────────────────────────────────────────┤
  │ d_model          │ 64           │ 注意力维度，影响全局。64→128 约 4x 参数量          │
  ├──────────────────┼──────────────┼────────────────────────────────────────────────────┤
  │ n_heads          │ 4            │ 注意力头数，需整除 d_model。影响表达力但不增加参数 │
  ├──────────────────┼──────────────┼────────────────────────────────────────────────────┤
  │ encoder_channels │ (64,128,256) │ UNet 层数和宽度。加层加宽都增参数                  │
  └──────────────────┴──────────────┴────────────────────────────────────────────────────┘

  推荐组合（按规模排列）：

  # ~4M (当前)
  d_model=64, encoder_channels=(64,128,256)

  # ~10M
  d_model=96, encoder_channels=(96,192,384)

  # ~16M
  d_model=128, encoder_channels=(128,256,512)

  # ~25M (加深一层)
  d_model=128, encoder_channels=(128,256,512,512)

  通过命令行调整的话，需要把这些参数暴露到 CLI。要我加一下吗？还是你直接改 src/configs/dpcaunet_config.py 里的值？
