 仿真数据生成 CUDA GPU 加速                                                                                                                                                                                                                                                                            背景                                                                                                                                                                                                                                                                                                  scripts/generate_data.py 的瓶颈在两个 CPU 密集操作：                                                                                               1. SolveForward (~50-150ms/样本): FEM 矩阵组装 + scipy.sparse.linalg.spsolve                                                                       2. reconstruct_list (~200-500ms/样本): 密集矩阵乘法 BJ.T @ GammaInv @ BJ + 5 次 np.linalg.solve                                                   

 按照 update1.md 思路，使用 CuPy 替换关键运算，并额外优化对角矩阵乘法。

 加速策略

 加速点 1: SolveForward — CuPy 密集求解

 文件: src/ktc_methods/KTCFwd.py L117

 FEM 组装循环 (L60-74) 是逐元素 Python 循环，保持不变。仅加速求解步骤：

 # 原始 (CPU sparse solve):
 UU = sp.sparse.linalg.spsolve(self.A, self.b)

 # GPU (转为密集后用 cuSOLVER):
 A_gpu = cp.asarray(self.A.toarray())  # ~6000×6000, ~290MB
 b_gpu = cp.asarray(self.b)
 UU = cp.asnumpy(cp.linalg.solve(A_gpu, b_gpu))

 用密集求解而非稀疏，因为：CuPy spsolve 不支持矩阵 RHS（76 列），而 6000×6000 密集矩阵 GPU 求解极快（几 ms）。

 加速点 2: reconstruct_list — 对角优化 + CuPy

 文件: src/reconstruction/linearised_reco.py L99-130

 关键优化: GammaInv 是对角矩阵，当前代码创建完整 M×M 密集矩阵再乘，浪费算力：

 # 原始 (创建 2356×2356 密集对角矩阵):
 GammaInv = np.diag(gamma_vec)                    # O(M²) 内存
 JGJ = BJ.T @ GammaInv @ BJ                      # O(M²N) 运算

 # 优化 (向量化行缩放, CPU 和 GPU 都受益):
 BJ_w = BJ * gamma_vec[:, None]                   # O(MN) 行缩放
 JGJ = BJ_w.T @ BJ                                # O(MN²) 矩阵乘
 b = BJ_w.T @ deltaU                              # O(MN) 向量乘

 GPU 模式下 BJ / Rtv / Rsm 预加载到 GPU，所有矩阵运算在 GPU 完成。

 加速点 3: interpolate_to_image — 缓存 Delaunay 三角化

 文件: src/reconstruction/linearised_reco.py L150-163

 当前每次调用 LinearNDInterpolator(self.pos, sigma) 都重建 Delaunay 三角化，但 self.pos（网格质心）不变。预计算一次缓存复用。

 修改文件清单

 1. src/ktc_methods/KTCFwd.py

 - __init__ 新增 use_gpu=False 参数
 - SolveForward L117：当 use_gpu=True 时用 CuPy 密集求解替换 spsolve
 - 向后兼容：默认 use_gpu=False，行为不变

 2. src/reconstruction/linearised_reco.py

 - __init__ 新增 use_gpu=False 参数
   - use_gpu=True 时将 BJ / Rtv / Rsm 预加载为 CuPy 数组
   - 预计算 Delaunay 三角化和像素网格坐标并缓存
 - reconstruct_list: 对角优化（CPU/GPU 都受益）+ CuPy 加速
   - GPU 路径: gamma_vec → BJ_w = BJ_gpu * gamma → JGJ = BJ_w.T @ BJ → solve on GPU
   - CPU 路径: 同样用向量化对角乘法替换完整对角矩阵（纯 CPU 也提速）
 - interpolate_to_image: 使用缓存的 Delaunay 三角化

 3. scripts/generate_data.py

 - 新增 --gpu 命令行参数
 - 当 --gpu 时：EITFEM(use_gpu=True) + LinearisedRecoFenics(use_gpu=True)
 - 新增逐步计时：phantom / forward / noise / reco / interp / io
 - 训练结束后保存计时摘要到 {output_dir}/timing_{mode}.json

 4. 新建 scripts/benchmark_data_gen.py

 独立基准测试脚本：
 - 用相同随机种子生成 N 个样本（默认 10），分别用 CPU 和 GPU
 - 分步记录耗时（phantom / forward / reco / total）
 - 打印对比表格 + 保存到 results/gpu_benchmark.json
 - 支持 --measurements-only 和完整模式

 # 用法
 python scripts/benchmark_data_gen.py --num-samples 10
 python scripts/benchmark_data_gen.py --num-samples 10 --measurements-only

 预期加速效果

 ┌──────────────────────────────────┬───────────┬───────────┬────────┐
 │               操作               │ CPU 耗时  │ GPU 预估  │ 加速比 │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ SolveForward (solve 部分)        │ 20-70ms   │ 2-5ms     │ 5-10×  │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ SolveForward (assembly, 不变)    │ 30-80ms   │ 30-80ms   │ 1×     │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ reconstruct_list (JGJ + 5×solve) │ 200-500ms │ 20-50ms   │ 5-10×  │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ interpolate_to_image ×5          │ 50-200ms  │ 10-40ms   │ 3-5×   │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ measurements-only 总计           │ 70-180ms  │ 45-100ms  │ ~2×    │
 ├──────────────────────────────────┼───────────┼───────────┼────────┤
 │ 完整模式总计                     │ 320-900ms │ 100-250ms │ ~3-5×  │
 └──────────────────────────────────┴───────────┴───────────┴────────┘

 注：FEM 组装循环是纯 Python，保持不变，是 SolveForward 的主要瓶颈。

 验证

 # 基准测试
 python scripts/benchmark_data_gen.py --num-samples 10
 python scripts/benchmark_data_gen.py --num-samples 10 --measurements-only

 # 实际使用
 python scripts/generate_data.py --level 1 --num-images 100 --measurements-only --gpu
 python scripts/generate_data.py --level 1 --num-images 100 --gpu

 关键文件

 ┌───────────────────────────────────────┬─────────────────────────────────────┐
 │                 文件                  │                操作                 │
 ├───────────────────────────────────────┼─────────────────────────────────────┤
 │ src/ktc_methods/KTCFwd.py             │ 修改: SolveForward 加 CuPy 密集求解 │
 ├───────────────────────────────────────┼─────────────────────────────────────┤
 │ src/reconstruction/linearised_reco.py │ 修改: 对角优化 + CuPy + 缓存三角化  │
 ├───────────────────────────────────────┼─────────────────────────────────────┤
 │ scripts/generate_data.py              │ 修改: --gpu + 逐步计时              │
 ├───────────────────────────────────────┼─────────────────────────────────────┤
 │ scripts/benchmark_data_gen.py         │ 新建: CPU vs GPU 基准测试           │
 └───────────────────────────────────────┴─────────────────────────────────────┘
