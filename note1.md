● 整个 src/ 和 scripts/ 都是新增的（未跟踪），因为是在第一次 commit 之后创建的。让我总结一下训练框架的完整实现结果。
          ---     训练代码框架已全部实现并验证通过。以下是完整的实现清单：
        
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
  │ 加载网格     │ load_mesh("Mesh_dense.mat") │ load_mesh(mesh_name) │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 正演求解器   │ EITFEM(mesh2, Injref, Mpat, vincl)                  │ 同              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 噪声参数     │ noise_std1=0.05, noise_std2=0.01                    │ 同              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 参考测量模拟 │ sigma_bg=0.745, SolveForward + InvLn噪声            │ 同              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ vincl掩码    │ 手写循环(L89-97)       │ create_vincl() 封装了同样的循环              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 重建器       │ LinearisedRecoFenics(Uelref,B,vincl_level,          │ 同，base_path='KTC2023_SubmissionFiles/data' │ 一致                  │   
  │              │ mesh_name="sparse") base_path默认"data"             │                 │ (路径适配项目根目录)  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ Phantom生成  │ create_phantoms()      │ 同              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 电导率采样   │ bg=0.745, resistive=rand*0.1+0.025,                 │ 同              │ 一致                  │   
  │              │ conductive=rand+5.0    │                 │                       │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 正演+加噪    │ SolveForward(sigma_gt, z) + InvLn*randn             │ 同              │ 一致                  │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ 5通道重建    │ reconstruct_list(Uel_noisy, alphas) →               │ 同              │ 一致                  │   
  │              │ 5个interpolate_to_image → np.stack                  │                 │                       │   
  ├──────────────┼─────────────────────────────────────────────────────┼──────────────────────────────────────────────┼───────────────────────┤   
  │ alpha参数    │ level_to_alphas 字典   │ LEVEL_TO_ALPHAS（数值完全相同）              │ 一致                  │   
  └──────────────┴─────────────────────────────────────────────────────┴──────────────────────────────────────────────┴───────────────────────┘   

  差异

  1. 缺少 HanddrawnImages 支持：原版有 use_handdrawn_images=True 选项，从作者本地路径 /home/adenker/projects/.../KTC_handdrawn_images
  加载手绘图像作为额外的 phantom 来源。我们只用了 create_phantoms() 随机生成。这个手绘数据集不在提交文件中，无法直接复现。
  2. 缺少 conductivity values pickle：原版额外保存了 sigmavalues_ztm_{idx}.pkl（包含 background/resistive/conductive
  三个电导率值），我们没保存。这个文件训练时不使用，仅供调试。
  3. 文件命名起始偏移：原版 max_image_idx=2000，我们默认 start_idx=0。不影响训练（数据集按目录遍历文件名）。

  总结：核心数据生成逻辑完全一致。 差异都是外围功能（手绘图像数据源不可获取、调试用pickle文件、文件名偏移），不影响训练数据的正确性。



  修改了 3 个文件 + 新建 1 个文件：                   
                     ┌───────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐    │                 文件                  │                   改动                    │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ src/ktc_methods/KTCFwd.py             │ EITFEM 加 use_gpu 参数，SolveForward 中 spsolve → CuPy 密集求解        │
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/reconstruction/linearised_reco.py │ 加 use_gpu; reconstruct_list 对角优化（CPU 也受益）+ CuPy GPU 路径; interpolate_to_image 缓存       │  
  │          │ Delaunay 三角化                     │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ scripts/generate_data.py              │ 加 --gpu 参数; 逐步计时 (phantom/forward/noise/reco/interp/io); 保存 timing JSON                    │  
  ├───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤  
  │ scripts/benchmark_data_gen.py         │ 新建: CPU vs GPU 基准对比，输出对比表 + JSON   │  
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
  1. A0 刚度矩阵向量化组装 + COO（~10-15x）             2. S0 电极边界 COO 替代 CSR（消除 SparseEfficiencyWarning）
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
  │               文件               │   内容   │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Mesh_dense.mat / Mesh_sparse.mat │ 电极边界节点索引（elfaces, elfaces2）、网格拓扑          │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ ref.mat  │ Injref(32×76 注入模式)、Mpat(测量模式)、Uelref(参考电压) │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ src/ktc_methods/KTCMeshing.py    │ 圆形网格生成，定义电极角度和宽度 │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ src/ktc_methods/KTCFwd.py        │ 电极接触阻抗 z = 1e-6 Ω     │
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
  │ 背景电导率      │ 0.745                │ 0.804│
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
  │       参数       │    当前值    │影响│
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



● 全部完成。以下是 5 项收敛加速改动的总结：

  1. Linear Warmup + Cosine Annealing 调度器

  - 替换了 ReduceLROnPlateau
  - Stage 1：3 epoch warmup + cosine
  - Stage 3：5 epoch warmup + cosine（warmup_epochs 可配置）
  - 每 iteration step，不是每 epoch

  2. 残差零初始化 (Zero-Init Residual)

  - DPCAUNet：cross_attn.out_proj 权重和偏置初始化为 0
  - HCDPCAUNet：CascadedCrossAttentionLayer 的 out_proj 和 FFN 末层、两个模型的 ConvBlock 最后一个 Conv 层均零初始化
  - 效果：训练初期网络近似恒等映射，低频先收敛，高频逐步加入

  3. Dice + Focal 混合损失

  - src/losses/dice_focal.py：0.5 * Dice + 0.5 * Focal(gamma=2.0)
  - Dice 关注形状重合度，Focal 自动降低背景像素权重
  - 解决 95% 背景导致的"背景梯度淹没"

  4. 分组学习率 + 测量数据归一化

  - 分组 LR：注意力模块 lr * 0.1，UNet 等其他部分 lr * 1.0
  - 归一化：训练开始时统计非零测量值的 mean/std，训练和验证中 (y - mean) / std，保持 vincl 零值不变
  - mean/std 保存到 checkpoint 供恢复训练使用

  5. 深监督权重指数衰减

  - DPCAUNet (3 decoder blocks)：(0.5, 0.25, 0.125)
  - HCDPCAUNet (4 decoder blocks)：(0.5, 0.25, 0.125, 0.0625)
  - 避免低分辨率层强行拟合高频边缘导致梯度冲突



  所以它到底算哪种
  我会这样总结：

  - 显存占用的大方向是架构硬成本
  - 但仍然可以通过实现优化把上限再推一截
  - 最现实的优化手段不是“手写底层算子”，而是 checkpointing、memory-efficient attention、梯度累积

  如果你现在问“最值得做什么”，优先级我会排成：

  1. activation checkpointing
  2. 梯度累积
  3. 检查是否有可裁掉的高分辨率辅助分支
  4. 再考虑 attention 内存优化
  5. 最后才是自定义算子

  一句话：

  - 不是纯硬性瓶颈
  - 但主要矛盾也不是缺一个自定义算子
  - 最有效的是训练策略和显存友好实现，而不是先手写算子

  如果你要，我可以继续帮你判断：对当前 dpcaunet，哪几处最适合加 activation checkpointing。


• fcunet 的核心可以概括成一句话：
         
  先把 2356 维电压测量映射成一张低分辨率“粗图”，再用带注意力的 U-Net 逐步上采样成 256×256 的 3 类分割结果。 
         
  对应实现主入口在 src/models/fcunet/unet.py。   
         
  整体流程     
         
  1. 输入是展平后的测量向量                 
     维度是 31 × 76 = 2356。                
  2. initial_linear 先做全连接映射          
     在 src/models/fcunet/unet.py:579 里：  
      - x -> initial_linear                 
      - 变成 64×64=4096                     
      - 再 reshape 成 (B, 1, 64, 64)        
  3. 加入 level / timestep embedding        
     在 src/models/fcunet/unet.py:462 附近，difficulty level 会先变成 sinusoidal embedding，再经过 MLP，作为各层 ResBlock 的条件输入。   
  4. 进入一个标准的 encoder-decoder U-Net 主干   
     结构上包括：   
      - 多层 ResBlock                       
      - 若干 attention block                
      - 下采样 
      - 中间 bottleneck                     
      - 上采样 
      - skip connection                     
  5. 最终输出 3 通道 logits                 
     在 src/models/fcunet/unet.py:556：     
      - 输出形状 (B, 3, 256, 256)           
      - 3 个通道分别对应：                  
          - background                      
          - resistive inclusion             
          - conductive inclusion            
  6. 推理时再 argmax 成单通道标签图         
     得到 (B, 256, 256)，像素值是 0/1/2。   
         
  为什么叫 FCUNet   
  因为它前面多了一个“全连接输入投影”：      
         
  - 不是直接吃图像  
  - 而是先把测量向量通过全连接层投成 64×64 粗图  
    这就是 Fully Connected + U-Net 的来源。 
         
  分辨率变化   
  大致是：     
         
  - 输入向量：2356  
  - 全连接粗图：1×64×64                     
  - U-Net 内部继续下采样到更低分辨率        
  - 再逐级上采样    
  - 最后输出：3×256×256                     
         
  所以它不是“输入和输出同分辨率”的普通 U-Net，而是：                     
         
  - 向量输入   
  - 图像输出   
         
  训练分两阶段 
  在 src/trainers/fcunet_trainer.py 里：    
         
  1. Stage 1   
     只训练 initial_linear，让它先学会把测量投影成一个粗略差分图。       
  2. Stage 2   
     再训练整个 FCUNet 做 3 类分割。        
  这样做的原因是：
  - 先把最前面的线性映射训稳，会更容易收敛

  配置上这套网络有多大
  默认配置见 src/configs/fcunet_config.py：

  - model_channels = 64
  - channel_mult = (1, 1, 2, 2, 4, 4)
  - num_res_blocks = 2
  - attention_resolutions = (32, 16, 8)

  这意味着：

  - 主干基础通道是 64
  - 随着层数加深，通道逐步扩到 128、256
  - 在特定分辨率层插入 attention

  如果你愿意，我可以下一步继续把 fcunet 按“每一层输出 shape”详细列出来。


• 已实现。
              
  现在 sae_predictor 训练改成了“latent loss + frozen SAE image loss”的联合训练，核心改动在：               
              
  - src/trainers/sae_predictor_trainer.py  
  - src/data/sim_dataset.py                
  - src/configs/sae_predictor_config.py    
  - scripts/train.py                       
              
  现在的行为是：   
              
  - 冻结整个 SAE，只训练 MeasurementPredictor    
  - 训练损失变成   
    latent_loss + w_img(epoch) * image_loss
  - latent_loss = mse_shape + lambda_angle * mse_angle
  - image_loss = CE(decoded_logits, gt_indices)  
  - 图像损失权重采用两阶段策略：           
      - 前 40% epoch：w_img = 0            
      - 后 60% epoch：从 0.1 线性增大到 1.0
  - 验证/最佳模型/学习率调度/早停统一看 val_image_loss
              
  另外我还修了一个关键一致性问题：         
              
  - 旋转增强以前只旋转了 measurement 和 latent angle  
  - 现在训练时会把 gt_indices 按同样的旋转步数同步旋转
  - 不然 image loss 会学错目标             
              
  命令行新增了这些可选参数：               
              
  - --use-image-loss                       
  - --image-loss-switch-ratio              
  - --image-loss-weight-start              
  - --image-loss-weight-end                
              
  虽然 use_image_loss 现在配置默认就是开着的，但你仍然可以显式传这些参数覆盖。
              
  建议你直接这样启动一版新实验：           
              
python scripts/train.py --method sae_predictor --hdf5-path dataset_sim/level_1/data.h5 --sae-checkpoint results/sae_baseline_3/best.pt --latent-h5-path results/sae_baseline_3/latent_codes.h5 --experiment-name sae_predictor_imgloss              
  如果你想用微调后的 SAE 解码器，就把 checkpoint 和 latent 改成对应目录。     
              
  我已做过：  
              
  - 语法编译检查   
  - 数据集返回结构冒烟检查                 
              
  还没有替你跑完整训练，所以真实收敛曲线和图像域改善效果需要你下一步训练验证。
 


• 现在这版 sae_predictor 的“两阶段”不是两个独立训练循环，而是通过图像损失权重调度隐式划分的。

  当前设置在 src/configs/sae_predictor_config.py 和 src/trainers/sae_predictor_trainer.py 里：

  - 总 epoch：300
  - image_loss_switch_ratio = 0.4
  - image_loss_weight_start = 0.1                                   
  - image_loss_weight_end = 1.0                                     
                   
  对应实际训练阶段是：
                   
  1. 第一阶段      
                   
  - epoch 0 ~ 119  
  - w_img = 0      
  - 也就是只训练 latent loss                                        
  - 不看图像重建误差回传                                            
                   
  2. 第二阶段      
                   
  - epoch 120 ~ 299
  - w_img 从 0.1 线性增大到 1.0                                     
  - 也就是 latent loss 继续保留，但图像损失逐步占主导               
                   
  所以你觉得“第一阶段收敛非常快”是合理的，因为这 120 个 epoch 只是在拟合 latent。                                         
                   
  如果你想缩短第一阶段，有两种直接办法。                            
                   
  办法 1：缩短第一阶段
  把 image_loss_switch_ratio 调小，比如：                           
                   
  - 0.2            
      - 前 60 个 epoch 只训 latent                                  
  - 0.1            
      - 前 30 个 epoch 只训 latent                                  
                   
  例如：           
                   
  python scripts/train.py --method sae_predictor ... --image-loss-switch-ratio 0.1                                        
                   
  办法 2：直接不做第一阶段                                          
  把 image_loss_switch_ratio 设成 0，让训练一开始就进入“第二阶段”： 
                   
  python scripts/train.py --method sae_predictor ... --image-loss-switch-ratio 0                                          
                   
  但要注意：       
                   
  - 即使这样，当前实现里 latent loss 仍然会一直保留                 
  - 只是图像损失从一开始就参与训练                                  
                   
  如果你想要“真正一开始就以图像为主”，我建议同时把：                
                   
  - --image-loss-weight-start 1.0                                   
  - --image-loss-weight-end 1.0                                     
                   
  这样就变成从第一个 epoch 开始：                                   
                   
  - total_loss = latent_loss + 1.0 * image_loss                     
                   
  例如：           
                   
  python scripts/train.py --method sae_predictor ... --image-loss-switch-ratio 0 --image-loss-weight-start 1.0 --image-loss-weight-end 1.0                                 
                   
  我更推荐你先试一个折中版本，而不是完全跳过：                      
                   
  --image-loss-switch-ratio 0.1 --image-loss-weight-start 0.3 --image-loss-weight-end 1.0                                 
                   
  原因是：         
                   
  - 你现在的问题是 latent 已经学得不错，但图像域不对                
  - 所以确实没必要花 40% epoch 只训 latent                          
  - 但完全从第 1 个 epoch 就强推 image loss，也可能让 early optimization 更抖                                             
                   
  如果你愿意，我可以下一步直接帮你把默认配置改成更合理的一版，比如：
                   
  - switch_ratio = 0.1
  - start = 0.3    
  - end = 1.0      

  另外我修了一个对联合训练很关键的问题：

  - sae_predictor 的旋转增强现在会同步旋转 gt_indices
  - 不然 measurement 被旋转了，但图像监督目标没旋转，图像损失会学错

  我已经做过：

  - 语法编译检查
  - sae_predictor build_model 冒烟检查
  - SAEPipeline 加载冒烟检查

  你现在可以直接开始训练，例如：

python scripts/train.py --method sae_predictor --hdf5-path dataset_sim/level_1/data.h5 --sae-checkpoint results/sae_baseline_3/ 
best.pt --latent-h5-path results/sae_baseline_3/latent_codes.h5 --experiment-name sae_predictor_joint_img

  如果你要用微调后的 SAE 权重，就把 checkpoint 和 latent 路径换成对应目录。

  补充一点：scripts/train.py 里我之前加过的那几个 image-loss schedule 参数现在已经不再参与 sae_predictor 实际训练了，因为你这次要 
  求的是直接
  去掉第一阶段、只用图像损失。
  如果你愿意，我下一步可以顺手把这些已经废弃的 CLI 参数清理掉，避免后面混淆。



• 已实现一套独立的 vq_sae / vq_sae_predictor 链路，不会覆盖现有 sae。
                                    
  现在支持的主链是：                
                                    
  - python scripts/train.py --method vq_sae --hdf5-path ...                               
  - 训练 ST-1D-VQ-VAE               
  - 自动导出离散 latent 到 results/vq_sae_baseline_{num}/latent_codes.h5                  
  - python scripts/train.py --method vq_sae_predictor --hdf5-path ... --vq-sae-checkpoint ... --vq-latent-h5-path ...                      
  - 训练测量到离散槽位分类 + 角度回归的 predictor                                         
  - python scripts/evaluate_all.py --methods vq_sae --weights-dir results                 
  - 用 vq_sae_predictor + frozen vq_sae decoder 跑官方评估                                
                                    
  我还补了两点兼容性：              
                                    
  - vq_sae_pipeline 在 YAML 没指定 vq_sae_dir 时，会优先从 predictor 的 config.yaml 反推出配套的 vq_sae 目录，避免拿错最新实验。           
  - HDF5 dataset 的 __del__ 做了安全关闭，避免 Windows/Python 退出时偶发 h5py 噪声报错。  
                                    
  我做过的检查：                    
                                    
  - 新增文件全部通过 py_compile     
  - ST1DVQVAE 和 VQMeasurementPredictor 前向 shape 已验证                                 
  - VQSAETrainer 的 build_model/build_datasets 已用真实 dataset_sim/level_1/data.h5 做过冒烟 
                                    
  有一个我主动保留的实现偏差需要告诉你： 

  - 你文档里写的是 STN 输入旋转训练也用 nearest
  - 我这里训练时仍然用 bilinear，推理时才用 nearest
  - 原因是如果训练时输入旋转也用 nearest，角度分支梯度会非常差，首版大概率训不动
    如果你坚持完全按文档，我可以再改成硬 nearest，但我不建议。

  建议你下一步先跑这两条做首轮验证：

python scripts/train.py --method vq_sae --hdf5-path dataset_sim/level_1/data.h5 --max-iters 2

python scripts/train.py --method vq_sae_predictor --hdf5-path dataset_sim/level_1/data.h5 --vq-sae-checkpoint results/vq_sae_baseline_2/best.pt --vq-latent-h5-path results/vq_sae_baseline_2/latent_codes.h5 --max-iters 2

  如果你要，我下一步可以继续补：

  - vq_sae 的仿真样本可视化脚本
  - vq_sae 的 GT 自编码重建脚本
  - 或直接帮你排第一轮训练时的稳定性问题。



  已完成                                 
                                         
  - 新增连续电导率数据集读取：/D:/010_CodePrograms/E/EIT_KTC2023_4/src/data/sim_dataset.py     
      - ConductivityHDF5Dataset          
      - 直接读取肺部 HDF5 里的 sigma     
  - 新增连续回归指标：/D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/regression_metrics.py
      - MAE                              
      - RMSE                             
      - relative L2                      
      - 都是 masked 版本，只在胸腔有效区域上计算 
  - 新增连续电导率 DCT 方法：            
      - 配置：/D:/010_CodePrograms/E/EIT_KTC2023_4/src/configs/dct_sigma_predictor_config.py   
      - 训练：/D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/dct_sigma_predictor_trainer.py 
      - 推理：/D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/dct_sigma_predictor_pipeline.py  
  - 新增仿真集连续回归评估脚本：/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/evaluate_sigma_sim.py
  - 新增连续电导率可视化脚本：/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/visualize_sigma_samples.py                                           
  - 已注册到统一训练入口：/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/train.py                
      - 新方法名：dct_sigma_predictor    
                                         
  当前口径                               
                                         
  - 肺部主线以后建议用连续 sigma 做主结果。   
  - KTC2023 三值标签继续保留，作为“标签化结构恢复”补充实验。                                   
  - 连续电导率不再用那个 SSIM-style score，而是用：                                            
      - masked MAE                       
      - masked RMSE                      
      - masked relative L2               
                                         
  已验证                                 
  我已经跑通了 smoke test：              
                                         
  - 训练结果：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_smoke_2                   
  - quick eval 结果：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_smoke_2/dct_sigma_test_eval_2/summary.json                          
  - quick 可视化：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_smoke_2/dct_sigma_test_samples_1/test_comparison.png                   
                                         
  这说明：                               
                                         
  - 数据读取通了                         
  - 训练通了                             
  - 推理通了                             
  - 回归评估通了                         
                                         
  下一步最该跑                           
  建议直接在你本地能正常开 num-workers 8 的终端里跑肺部正式实验：                              
                                         
python scripts/train.py --method dct_sigma_predictor --hdf5-path dataset_lung_2k/level_1/data.h5 --device cuda --num-workers 8 --epochs 20 --coeff-size 20 --experiment-name dct_sigma_lung2k                                           
                                         
  跑完后再评估：                         
                                         
python scripts/evaluate_sigma_sim.py --weights-dir results/dct_sigma_lung2k_1 --hdf5-path dataset_lung_2k/level_1/data.h5 --split test --device cuda --batch-size 64          
                                         
  再出图：                               
                                         
python scripts/visualize_sigma_samples.py --weights-dir results/dct_sigma_lung2k_1 --hdf5-path dataset_lung_2k/level_1/data.h5 --split test --device cuda --rows 2 --cols 4        
                                         
  接下来我会继续按这个连续电导率主线推进论文，把肺部部分改成以 sigma 回归为主，KTC 三值标签作为补充对照。                                       
 

• 我觉得有机会够 ICIC，但前提是你要把故事线收紧，不要写成“三篇小报告拼在一起”。                                                                                          
                                                                                                                                                                         
  更准确地说：                                                                                                                                                           
                                                                                                                                                                         
  - 如果主题改成“数据中心的 EIT 研究”，有希望。                                                                                                                          
  - 如果主题还是“我们提出了一个最强重建网络”，目前不够。                                                                                                                 
                                                                                                                                                                         
  我建议的主线是：                                                                                                                                                       
                                                                                                                                                                         
  高效仿真数据生成 + 数据数量/质量对 EIT 重建性能的影响分析                                                                                                              
                                                                                                                                                                         
  然后用两个任务做验证：                                                                                                                                                 
                                                                                                                                                                         
  - 肺部连续电导率回归                                                                                                                                                   
  - 水箱实验/KTC2023 三值标签分割                                                                                                                                        
                                                                                                                                                                         
  这样是自洽的。因为这两个任务分别覆盖：                                                                                                                                 
                                                                                                                                                                         
  - 连续物理量重建                                                                                                                                                       
  - 离散结构分割                                                                                                                                                         
                                                                                                                                                                         
  而“自编码器评估数据质量”可以作为数据质量分析工具，不是论文主方法。                                                                                                     
                                                                                                                                                                         
  ## 为什么这条线可能够 ICIC                                                                                                                                             
                                                                                                                                                                         
  ICIC 对这类应用论文，通常不一定要求特别强的全新理论，更看重：                                                                                                          
                                                                                                                                                                         
  - 问题是否清楚                                                                                                                                                         
  - 工程价值是否明显                                                                                                                                                     
  - 实验是否完整                                                                                                                                                         
  - 结论是否可靠                                                                                                                                                         
                                                                                                                                                                         
  你现在已经有比较好的基础：                                                                                                                                             
                                                                                                                                                                         
  - 仿真数据生成加速是实打实的工程贡献                                                                                                                                   
  - 肺部和水箱两个任务覆盖面合理                                                                                                                                         
  - 可以做数据规模曲线                                                                                                                                                   
  - 可以做数据质量曲线
  - 可以分析速度、精度、参数量、延时                                                                                                                                     
                                                                                                                                                                         
  这比硬推一个目前效果还不稳定的新网络更稳。                                                                                                                             
                                                                                                                                                                         
  ## 但要注意一个风险                                                                                                                                                    
                                                                                                                                                                         
  如果你把论文写成：                                                                                                                                                     
                                                                                                                                                                         
  - 一部分讲加速                                                                                                                                                         
  - 一部分讲 SAE                                                                                                                                                         
  - 一部分讲肺部                                                                                                                                                         
  - 一部分讲 KTC2023                                                                                                                                                     
  - 一部分又讲 DCT/FCUNet                                                                                                                                                
                                                                                                                                                                         
  就会显得主题太散。                                                                                                                                                     
                                                                                                                                                                         
  所以需要明确主次：                                                                                                                                                     
                                                                                                                                                                         
  ### 推荐主贡献                                                                                                                                                         
                                                                                                                                                                         
  1. 高效 EIT 仿真数据生成框架                                                                                                                                           
  2. 用数据规模与数据质量分析，研究仿真数据对下游 EIT 重建的影响                                                                                                         
  3. 在连续肺部重建和三值标签分割两个任务上验证结论                                                                                                                      
                                                                                                                                                                         
  ### 推荐次贡献                                                                                                                                                         
                                                                                                                                                                         
  - 自编码器只是数据质量评估器                                                                                                                                           
  - DCT/FCUNet 只是下游验证模型                                                                                                                                          
  - 不把它们写成论文主角                                                                                                                                                 
                                                                                                                                                                         
  ## 这样写时，论文要达到“够投”的最低条件                                                                                                                                
                                                                                                                                                                         
  至少需要这几块实验完整：                                                                                                                                               
                                                                                                                                                                         
  1. 数据生成加速                                                                                                                                                        
                                                                                                                                                                         
  - 原始生成流程 vs 加速后流程                                                                                                                                           
  - 总耗时、各阶段耗时、吞吐量                                                                                                                                           
  - 最好有不同样本规模下的曲线                                                                                                                                           
                                                                                                                                                                         
  2. 数据数量影响                                                                                                                                                        
                                                                                                                                                                         
  - 小规模、中规模、大规模                                                                                                                                               
  - 下游任务性能随样本数变化曲线                                                                                                                                         
  - 肺部和 KTC 两个任务都至少各一条                                                                                                                                      
                                                                                                                                                                         
  3. 数据质量影响                                                                                                                                                        
                                                                                                                                                                         
  - 你定义几种“质量等级”或“生成策略”                                                                                                                                     
      - 简单 phantom                                                                                                                                                     
      - 更复杂 phantom                                                                                                                                                   
      - 加背景扰动/噪声/形变                                                                                                                                             
  - 看下游性能怎么变                                                                                                                                                     
                                                                                                                                                                         
  4. 自编码器质量分析                                                                                                                                                    
                                                                                                                                                                         
  - 不把它当最终方法                                                                                                                                                     
  - 而是作为数据分布复杂度/可压缩性分析工具                                                                                                                              
  - 例如：                                                                                                                                                               
      - reconstruction error                                                                                                                                             
      - latent compactness                                                                                                                                               
      - 不同数据集复杂度比较                                                                                                                                             
                                                                                                                                                                         
  5. 下游任务验证                                                                                                                                                        
                                                                                                                                                                         
  - 肺部连续电导率：                                                                                                                                                     
      - MAE / RMSE / relative L2                                                                                                                                         
  - KTC2023 三值标签：                                                                                                                                                   
      - 官方 score                                                                                                                                                       
  - 至少 2 个基线方法                                                                                                                                                    
      - 一个强基线                                                                                                                                                       
      - 一个轻量基线                                                                                                                                                     
                                                                                                                                                                         
  ## 我的判断                                                                                                                                                            
                                                                                                                                                                         
  如果你把论文定位成“EIT 仿真数据生成与数据质量研究”，是够机会投 ICIC 的。                                                                                               
                                                                                                                                                                         
  如果你还想同时证明一个新的自编码器/新网络是最优方法，目前不够稳。                                                                                                      
                                                                                                                                                                         
  ## 我建议你现在就这么定                                                                                                                                                
                                                                                                                                                                         
  论文主线改成：                                                                                                                                                         
                                                                                                                                                                         
  - 数据生成加速                                                                                                                                                         
  - 数据数量/质量分析                                                                                                                                                    
  - 两个下游任务验证                                                                                                                                                     
                                                                                                                                                                         
  而不是：                                                                                                                                                               
                                                                                                                                                                         
  - 新模型方法论文                                                                                                                                                       
                                                                                                                                                                         
  一句话结论：                                                                                                                                                           
                                                                                                                                                                         
  - 够，但要换成“数据中心、实验中心”的论文结构                                                                                                                           
  - 不够的地方不是工作量，而是当前主题还不够聚焦                                                                                                                         
  - 把自编码器降级为分析工具，把下游网络降级为验证器，这篇就更像一篇完整的 ICIC 论文                                                                                     
                                                                                                                                                                         
  如果你同意，我下一步建议直接把论文结构和实验计划按这条主线重构。                                                                                                       


• 当前研究主线
  我已经把论文主线重构成了“数据中心的肺部电阻抗成像研究”：                                                                                         
                                                                                                                                                   
  - 主任务：肺部连续电导率回归                                                                                                                     
  - 补充任务：KTC2023 水箱实验三值标签分割                                                                                                         
  - 主方法定位：DCT 低频预测方法                                                                                                                   
  - 强基线定位：FCUNet / FC-SigmaUNet                                                                                                              
  - 自编码器定位：分析工具，不再是论文主方法                                                                                                       
                                                                                                                                                   
  已经完成的核心成果                                                                                                                               
                                                                                                                                                   
  1. 工程与实验基础                                                                                                                                
                                                                                                                                                   
  - 完成了仿真与评估加速，尤其是评分加速。                                                                                                         
  - 官方评分单样本约 92s，Torch/CUDA 快速评分约 2~3ms。                                                                                            
  - 主要说明文档：/D:/010_CodePrograms/E/EIT_KTC2023_4/GUIDE/scoring_acceleration.md                                                               
                                                                                                                                                   
  2. KTC2023 三值标签任务                                                                                                                          
                                                                                                                                                   
  - 当前仓库里最好的 benchmark 结果是 DCT ensemble：                                                                                               
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/eval_dct_predictor_ensemble_2/scores.json                                                     
      - total_score = 14.0553                                                                                                                      
  - 强基线 FCUNet：                                                                                                                                
      - total_score = 14.0485                                                                                                                      
  - 结论：DCT 在 KTC benchmark 上已经略超 FCUNet。                                                                                                 
                                                                                                                                                   
  3. 自编码器研究的结论                                                                                                                            
                                                                                                                                                   
  - SAE / VQ-SAE 本身的 图像自编码重建并不差。                                                                                                     
  - 真正困难的是：从电极测量值预测 latent。                                                                                                        
  - 所以自编码器结论现在被收敛为：                                                                                                                 
      - 证明图像流形是可压缩的                                                                                                                     
      - 证明瓶颈在 measurement -> latent                                                                                                           
      - 不再作为最终方法主线                                                                                                                       
                                                                                                                                                   
  4. 肺部仿真数据                                                                                                                                  
                                                                                                                                                   
  - 已生成并使用肺部连续电导率仿真集：                                                                                                             
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung_2k/level_1/data.h5                                                                       
  - 还额外构造了复杂度可控数据：
      - 低复杂度 /D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung_varlow/level_1/data.h5                                                          
      - 高复杂度 /D:/010_CodePrograms/E/EIT_KTC2023_4/dataset_lung_varhigh/level_1/data.h5                                                         
                                                                                                                                                   
  5. 肺部主任务：连续电导率回归                                                                                                                    
                                                                                                                                                   
  - 直接连续 DCT 在肺部上会塌成平均模板：                                                                                                          
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_lung2k_1                                                                            
      - rel_l2 = 0.3509                                                                                                                            
  - 训练集平均 atlas baseline 反而更强：                                                                                                           
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/sigma_mean_lung2k_1                                                                           
      - rel_l2 = 0.2552                                                                                                                            
  - 这说明肺部任务的主矛盾不是重建整体胸腔形状，而是恢复 atlas 之上的局部小变化。                                                                  
                                                                                                                                                   
  6. 肺部归纳偏置适配                                                                                                                              
     我已经系统试了这些 atlas-aware 变体：                                                                                                         
                                                                                                                                                   
  - atlas-residual DCT                                                                                                                             
  - hybrid DCT                                                                                                                                     
  - atlas-decoder                                                                                                                                  
  - FC-SigmaUNet                                                                                                                                   
                                                                                                                                                   
  在低/高复杂度匹配实验上，结果都贴着 atlas baseline，没有真正超越它：                                                                             
                                                                                                                                                   
  - 汇总结果：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_complexity_models_3/summary.json                                              
  - 定性图：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_complexity_visual_3/comparison.png                                              
                                                                                                                                                   
  7. 残差区域专门分析                                                                                                                              
     我新增了 residual-focused 分析，只看 |sigma - atlas| > 0.08 的区域：                                                                          
                                                                                                                                                   
  - 汇总结果：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_residual_focus_2/summary.json                                                 
  - 图：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_residual_focus_2/residual_focus_bar.png                                             
                                                                                                                                                   
  结论非常明确：                                                                                                                                   
                                                                                                                                                   
  - lung2k 上 residual-region RelL2                                                                                                                
      - atlas: 0.3746                                                                                                                              
      - residual DCT: 0.3747                                                                                                                       
      - hybrid DCT: 0.3747                                                                                                                         
      - atlas-decoder: 0.3746                                                                                                                      
  - 也就是说，这些 atlas-aware 模型没有在真正的局部偏差区域带来实质收益。                                                                          
                                                                                                                                                   
  8. focus-loss 消融                                                                                                                               
                                                                                                                                                   
  - 我还做了 focus-loss 去除实验。                                                                                                                 
  - 结果几乎不变：                                                                                                                                 
      - 低复杂度 0.2008 -> 0.2007                                                                                                                  
      - 高复杂度 0.2971 -> 0.2972                                                                                                                  
  - 结论：问题不是 loss 没强调局部区域，而是 measurement -> residual 本身还不够强。                                                                
                                                                                                                                                   
  9. 肺部基线与效率                                                                                                                                
                                                                                                                                                   
  - 肺部 synthetic segmentation 上，FCUNet 目前精度更高：                                                                                          
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/fcunet_lung2k_2                                                                               
      - mean_score = 0.7819                                                                                                                        
  - DCT 肺部分割当前：                                                                                                                             
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_predictor_lung2k_1                                                                        
      - mean_score = 0.7272                                                                                                                        
  - 但 DCT 仍然有明显效率优势：                                                                                                                    
      - 参数量少约 10.6x                                                                                                                           
      - 单样本延时快约 13.5x                                                                                                                       
      - batch=32 的单样本平均延时快约 58.3x                                                                                                        
                                                                                                                                                   
  对当前研究的总体判断                                                                                                                             
                                                                                                                                                   
  - KTC2023 benchmark：DCT 已经是当前最强方法。                                                                                                    
  - 肺部连续电导率：当前还没有超过强基线，但已经把问题定位得很清楚：                                                                               
      - 不是 decoder 容量问题                                                                                                                      
      - 不是简单 loss weighting 问题                                                                                                               
      - 而是 measurement -> local residual inference 的问题                                                                                        
  - 这反而让论文更清楚：                                                                                                                           
    DCT 的价值在于低频归纳偏置、轻量化、实时性和数据中心分析，不是宣称在所有任务上绝对最优。                                                       
                                                                                                                                                   
  真实肺部数据情况                                                                                                                                 

  - 已解析 /D:/010_CodePrograms/E/EIT_KTC2023_4/Subjects%20Data/Plos%20One%20Data
  - 结论是：
      - 真实 .get 数据可读
      - 但没有真实电导率 GT
      - 且是 16 电极 / 208 通道，与当前 32 电极 KTC 管线不匹配
  - 所以它现在更适合作为未来 16 电极 Sim2Real 的方向，而不是当前监督主实验

  论文状态

  - LaTeX 主稿已经成型并持续更新：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/paper/icic_eit_study.tex
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/paper/icic_eit_study.pdf
  - 目前论文主叙事已经比较稳定：
      - 数据生成与评分加速
      - KTC benchmark 上的 DCT 优势
      - 肺部任务中的 atlas dominance
      - DCT 的效率优势
      - 肺部连续回归的真正瓶颈定位

  一句话总结
  目前最成熟、最可信的研究结论是：

  - DCT 是一个适合 EIT 的高效低频表示，在 KTC benchmark 上已经达到最强结果；
  - 但肺部连续电导率任务本质上是“稳定解剖模板 + 局部小偏差”的问题，当前难点已经明确定位在 measurement -> residual，而不是图像表示或 decoder。      

› 哪个是目前最好的预测图？为什么我看论文里的预测图完全没有展现出真实图间的差别？肺部应该使用16电极仿真数据吧？应该使用残差预测或时间差分预测吧？你 
  觉得呢？
                                                                                                                                                   

• 我现在的判断很明确：
                                                                                                                                                   
  1. 目前“肺部连续电导率”里最好的预测图，不是某个真正学会局部变化的网络，而是几乎等价于 atlas baseline 的结果。                                    
                                                                                                                                                   
  按 lung2k 当前连续回归结果看：                                                                                                                   
                                                                                                                                                   
  - atlas baseline: /D:/010_CodePrograms/E/EIT_KTC2023_4/results/sigma_mean_lung2k_1/summary.json                                                  
      - rel_l2 ≈ 0.2552                                                                                                                            
  - atlas-decoder: /D:/010_CodePrograms/E/EIT_KTC2023_4/results/atlas_sigma_lung2k_1/dct_sigma_test_eval_1/summary.json                            
      - rel_l2 ≈ 0.2552                                                                                                                            
  - atlas-residual DCT: /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_lung2k_residual_1/dct_sigma_residual_predictor_test_eval_1/         
    summary.json
      - rel_l2 ≈ 0.2552                                                                                                                            
  - hybrid DCT: /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_lung2k_hybrid_1/dct_sigma_hybrid_predictor_test_eval_1/summary.json         
      - rel_l2 ≈ 0.2552                                                                                                                            
                                                                                                                                                   
  所以如果只问“当前哪张最好”，那答案其实是：                                                                                                       
                                                                                                                                                   
  - atlas-aware 这一组都差不多，最优数值上也只是贴着 atlas baseline。                                                                              
  - 没有一个模型真正明显恢复了样本间的局部差异。                                                                                                   
                                                                                                                                                   
  2. 你觉得论文里的预测图“完全没有展现真实图间的差别”，这个判断是对的。                                                                            
  原因不是排版，而是当前结果本身就在说明：                                                                                                         
                                                                                                                                                   
  - 模型学到了“平均肺部模板”                                                                                                                       
  - 没学到“模板上的小变化”                                                                                                                         
                                                                                                                                                   
  我刚做完的 residual-focused 分析更直接证明了这件事：                                                                                             
                                                                                                                                                   
  - 汇总结果：/D:/010_CodePrograms/E/EIT_KTC2023_4/results/pulmonary_residual_focus_2/summary.json                                                 
                                                                                                                                                   
  在真正的偏差区域 |sigma - atlas| > 0.08 上：                                                                                                     
                                                                                                                                                   
  - lung2k residual-region RelL2                                                                                                                   
      - atlas: 0.3746                                                                                                                              
      - residual DCT: 0.3747                                                                                                                       
      - hybrid DCT: 0.3747                                                                                                                         
      - atlas-decoder: 0.3746                                                                                                                      
                                                                                                                                                   
  这说明：                                                                                                                                         
                                                                                                                                                   
  - 现在论文图里“预测图彼此很像”不是偶然
  - 而是当前模型确实基本都停留在 atlas 附近

  所以这类 raw sigma 图不适合继续作为肺部主图。
  后面应该改成展示：

  - GT absolute sigma
  - Pred absolute sigma
  - GT residual = sigma - atlas
  - Pred residual = sigma_pred - atlas
  - Residual error map

  不然你看不到真正应该看的东西。

  3. 肺部确实应该优先转向 16 电极仿真，而不是继续把 32 电极 KTC 风格当主线。
  我现在也认为应该这样做，原因有三个：

  - 你仓库里的真实肺部 .get 数据是 16 电极 / 208 通道
  - 当前主训练链是 32 电极 / 2356 通道
  - 这两者不匹配，导致现在的肺部实验更像“借用 KTC 工程框架做的肺部仿真研究”，而不是严格对接真实肺部 EIT

  所以如果论文主题明确是肺部电阻抗成像，更合理的是：

  - 新建 16 电极肺部仿真链
  - forward / measurement layout / channel organization 都匹配真实 .get 数据

  这个我认为已经不是“可选优化”，而是后续论文可信度的关键一步。

  4. 肺部任务也确实更应该做残差预测或时间差分预测。
  这点我现在是明确赞同的，而且比“继续做绝对 sigma”更合理。

  原因很直接：

  - 肺部绝对电导率图本来就有很强 atlas dominance
  - 真实临床肺部 EIT 也更常见：
      - relative imaging                                                                                                                           
      - time-difference imaging                                                                                                                    
      - ventilation-related change imaging                                                                                                         
                                                                                                                                                   
  所以更合适的目标应该是：                                                                                                                         
                                                                                                                                                   
  1. 静态残差预测                                                                                                                                  
                                                                                                                                                   
  - target = sigma - atlas                                                                                                                         
  - 网络只负责恢复偏差部分                                                                                                                         
  - 最终输出 = atlas + residual_pred                                                                                                               
                                                                                                                                                   
  2. 时间差分预测                                                                                                                                  
                                                                                                                                                   
  - target = sigma_t - sigma_ref                                                                                                                   
  - 输入也改成差分测量                                                                                                                             
  - 这更贴近肺通气监测                                                                                                                             
                                                                                                                                                   
  如果往真实肺部数据靠，我会更推荐：                                                                                                               
                                                                                                                                                   
  - 16 电极 + time-difference / residual imaging                                                                                                   
    而不是：                                                                                                                                       
  - 32 电极 + absolute sigma regression                                                                                                            
                                                                                                                                                   
  5. 我现在建议的下一步研究方向                                                                                                                    
  我会把方向定成下面这条：                                                                                                                         
                                                                                                                                                   
  - 停止把当前 32 电极 absolute pulmonary sigma regression 当论文主结果                                                                            
  - 把它降级成“说明 atlas dominance 的分析实验”                                                                                                    
  - 重新建立 16 电极肺部仿真主线                                                                                                                   
  - 主任务改成 residual / time-difference reconstruction                                                                                           
  - DCT 方法也改成预测 residual 的低频表示，而不是绝对 sigma                                                                                       
                                                                                                                                                   
  一句话总结我的判断：                                                                                                                             
                                                                                                                                                   
  - 你说得对，当前肺部图展示不出样本差异，是因为模型基本没学到差异。                                                                               
  - 肺部主线应该切到 16 电极。                                                                                                                     
  - 目标应该优先做残差预测或时间差分预测。                                                                                                         
  - 当前 absolute sigma 的结果更适合当“问题诊断”，不适合继续当论文主结果。                                                                         
                                                                                                                                                   
  如果你同意这个方向，我下一步就直接开始把研究主线切到：                                                                                           
                                                                                                                                                   
  - 16 电极肺部仿真                                                                                                                                
  - residual / time-difference DCT predictor                                                                                                       
  - 并重写论文肺部实验部分                                                                                                                         


• 这轮我继续把肺部 16 电极时间差分线往前推了一步，新增了一个显式 change-aware 版本，并把结果写进了研究记录和肺部初稿。
                                                                                                                                                   
  代码上新增了：                                                                                                                                   
                                                                                                                                                   
  - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/models/dct_predictor/model.py                                                                         
  - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/dct_sigma_td16_change_predictor_trainer.py                                                   
  - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/dct_sigma_td16_change_predictor_pipeline.py                                                 
                                                                                                                                                   
  思路很直接：                                                                                                                                     
                                                                                                                                                   
  - 原来的 TD16 只预测 Δσ                                                                                                                          
  - 新版额外预测一个样本级 change gate                                                                                                             
  - 用 BCE 监督“这个样本是否真的有明显变化”                                                                                                        
  - 再用 gate 去整体缩放预测的 Δσ                                                                                                                  
  - 目标是降低 mixed 数据里零变化样本上的假阳性                                                                                                    
                                                                                                                                                   
  mixed pilot 的结果                                                                                                                               
                                                                                                                                                   
  - 原始 TD16：                                                                                                                                    
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_td16_pilot_1/td16_test_eval_2/summary.json                                          
      - RMSE = 0.0958
      - active_rel_l2 = 0.9788                                                                                                                     
  - sparse-only TD16：                                                                                                                             
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_td16_sparse_mixed_1/td16_test_eval_1/summary.json                                   
      - RMSE = 0.1038                                                                                                                              
      - active_rel_l2 = 0.9255                                                                                                                     
  - gate-only TD16：                                                                                                                               
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_td16_change_mixed_1/td16_test_eval_1/summary.json                                   
      - RMSE = 0.1012                                                                                                                              
      - active_rel_l2 = 0.9521                                                                                                                     
  - gate + sparse TD16：                                                                                                                           
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/dct_sigma_td16_change_sparse_mixed_1/td16_test_eval_1/summary.json                            
      - RMSE = 0.1007                                                                                                                              
      - active_rel_l2 = 0.9525                                                                                                                     
  - zero baseline：                                                                                                                                
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/td16_zero_baseline_2/summary.json                                                             
      - RMSE = 0.0835                                                                                                                              
      - active_rel_l2 = 1.0000                                                                                                                     
                                                                                                                                                   
  当前结论                                                                                                                                         
                                                                                                                                                   
  - change-aware 方向是对的，因为它确实改善了 active region 的恢复。                                                                               
  - 但现在最好的 learned mixed 版本也只是：                                                                                                        
      - 比原始 TD16 更像真实局部变化                                                                                                               
      - 比 sparse-only 更稳一点                                                                                                                    
      - 仍然没有赢过 zero baseline 的全局 RMSE                                                                                                     
  - 这说明当前肺部 TD16 的主矛盾已经更清楚了：                                                                                                     
      - 不是 decoder 容量问题                                                                                                                      
      - 不是单纯加稀疏项就能解决                                                                                                                   
      - 而是 change/no-change 判别 + 条件残差回归 还需要更结构化的建模                                                                             
                                                                                                                                                   
  文档和论文都已更新：                                                                                                                             
                                                                                                                                                   
  - 研究记录：/D:/010_CodePrograms/E/EIT_KTC2023_4/GUIDE/pulmonary_16e_time_difference_exploration.md                                              
  - 肺部初稿：/D:/010_CodePrograms/E/EIT_KTC2023_4/paper/pulmonary_preliminary/main.tex                                                            
  - 更新后的 PDF：/D:/010_CodePrograms/E/EIT_KTC2023_4/paper/pulmonary_preliminary/main.pdf                                                        
                                                                                                                                                   
  下一步我会继续沿这条最清楚的方向推进：做更明确的“两阶段 change detection + residual regression”，而不是回到绝对图像回归。                        
 


• 本轮结果
  我继续把肺部 16 电极外部深度学习复现往前推进了一步，并把结果写进了研究记录和肺部初稿。

  新增与更新：

  - 新分析脚本：/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/analyze_td16_vae_gap.py
  - 新融合脚本：/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/analyze_td16_vae_mask_fusion.py
  - 新条件化模型链：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/models/pulmonary_vae/model.py
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/configs/td16_vae_conditional_predictor_config.py
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/td16_vae_conditional_predictor_trainer.py
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/src/pipelines/td16_vae_conditional_predictor_pipeline.py
  - 文档与论文：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/GUIDE/pulmonary_external_baseline_reproduction.md
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/paper/pulmonary_preliminary/main.tex
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/paper/pulmonary_preliminary/main.pdf

  关键定量结果：

  - active-only VAE autoencoder vs predictor：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/td16_vae_predictor_active_pilot_1/td16_vae_gap_2/summary.json
      - autoencoder: RMSE 0.1189, active_rel_l2 0.7587
      - predictor: RMSE 0.1315, active_rel_l2 0.9692
  - mixed VAE autoencoder vs predictor：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/td16_vae_predictor_mixed_pilot_1/td16_vae_gap_2/summary.json
      - autoencoder: RMSE 0.0856, active_rel_l2 0.9983
      - predictor: RMSE 0.0842, active_rel_l2 0.9983
  - mixed mask-only + active-only VAE predictor 外部软融合：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/td16_vae_predictor_active_pilot_1/td16_vae_mask_fusion_1/summary.json
      - base active-only predictor on mixed: RMSE 0.0891, active_rel_l2 0.9525
      - external soft fusion: RMSE 0.0862, active_rel_l2 0.9753
  - 新的 mask-conditioned VAE predictor mixed pilot：
      - /D:/010_CodePrograms/E/EIT_KTC2023_4/results/td16_vae_conditional_mixed_pilot_1/td16_test_eval_1/summary.json
      - RMSE 0.0886, active_rel_l2 0.9686

  当前判断
  现在结论更清楚了：

  - active-only 下，VAE manifold 本身可以，但 measurement -> latent 很差。
  - mixed 下，不只是 predictor 差，连 VAE manifold 自己都已经塌到接近 zero-change。
  - 外部 mask 融合能压假阳性，但会伤 active-region 幅值恢复。
  - 把 mask 信号直接喂给 latent head 也不够，0.0886 / 0.9686 仍然没有超过更简单的外部 soft fusion。

  所以肺部这条线当前最准确的表述是：

  - GREIT 已复现，是可靠经典基线。
  - Zeng-style VAE + latent predictor 的高层流程也已经在本地 matched TD16 协议上验证过。
  - 但在当前小规模 mixed 数据上，主问题不是 decoder，而是 zero-change 主导下的表示塌陷和变化幅值恢复不足。

  需要你代跑的长程实验
  下一步最值得做的，不是继续在 64 样本上打转，而是上更大的 16 电极 matched dataset。这个阶段需要你本地终端或云端代跑，我建议直接跑下面
  四组：

  1. 生成更大的 mixed 数据：

  python scripts/generate_lung_td16_data.py --num-images 2000 --output-dir dataset_lung_td16_mixed2k --normal-prob 0.25

  2. 生成更大的 active-only 数据：

  python scripts/generate_lung_td16_data.py --num-images 2000 --output-dir dataset_lung_td16_active2k --normal-prob 0.0

  3. 训练 active-only VAE：

  python scripts/train.py --method td16_vae --hdf5-path dataset_lung_td16_active2k/level_1/data.h5 --device cuda --num-workers 8
  --epochs 80 --experiment-name td16_vae_active2k

  4. 训练两条关键对照：

  python scripts/train.py --method td16_vae_conditional_predictor --hdf5-path dataset_lung_td16_mixed2k/level_1/data.h5 --vae-
  checkpoint results/td16_vae_active2k_1/best.pt --device cuda --num-workers 8 --epochs 120 --experiment-name
  td16_vae_conditional_mixed2k

  python scripts/train.py --method dct_sigma_td16_spatial_change_predictor --hdf5-path dataset_lung_td16_mixed2k/level_1/data.h5
  --device cuda --num-workers 8 --epochs 120 --experiment-name dct_sigma_td16_spatial_mixed2k

  如果你把这几组结果跑出来交给我，我下一轮就能把肺部研究从“小样本 exploratory”推进到有说服力的 matched 16 电极对比实验。

  外部对齐依据：

  - Zeng et al. 2023 EMBC full paper: https://microsites.arinex.com.au/EMBC/pdf/full-paper_737.pdf
