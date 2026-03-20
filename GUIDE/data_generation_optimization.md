# EIT 仿真数据生成性能优化总结

## 概览

EIT 仿真数据生成的每个样本包含两个计算密集步骤：

1. **SolveForward**：FEM 刚度矩阵组装 + 稀疏线性方程组求解
2. **reconstruct_list**：5 组正则化参数的线性化重建

优化手段包括：NumPy 向量化、COO 稀疏组装、几何/边界条件预计算与缓存、CuPy GPU 加速、对角矩阵算法优化、线性化重建投影矩阵预计算。

### Benchmark 结果（RTX 5070 Ti Laptop GPU, 5 samples, full mode）

**第一轮优化 [A-G]**：

| 步骤 | 优化前 CPU | 优化前 GPU | 旧加速比 | 优化后 CPU | 优化后 GPU | 新加速比 |
|------|-----------|-----------|---------|-----------|-----------|---------|
| forward | 1026ms | 6849ms | 0.1x | **334ms** | **348ms** | 1.0x |
| reco | 22789ms | 3353ms | 6.8x | 22801ms | **3231ms** | **7.1x** |
| interp | 93ms | 60ms | 1.6x | 69ms | 63ms | 1.1x |
| **total** | **23913ms** | **10266ms** | **2.3x** | **23210ms** | **3646ms** | **6.4x** |

**第二轮优化 [H]**（投影矩阵预计算）— `results/gpu_benchmark_1.json`：

| 步骤 | [A-G] CPU | [A-G] GPU | [A-H] CPU | 加速比 (vs [A-G] GPU) |
|------|-----------|-----------|-----------|----------------------|
| forward | 334ms | 348ms | **349ms** | 1.0x |
| reco | 22801ms | 3231ms | **32ms** | **101x** |
| interp | 69ms | 63ms | **73ms** | 0.9x |
| **total** | **23210ms** | **3646ms** | **458ms** | **8.0x** |

优化 H 使 reco 从 3231ms 降至 32ms（101x 加速），**GPU 已无必要**——CPU 单独即可达到 458ms/样本。当前瓶颈完全转移至 forward 的 CPU 稀疏求解（349ms，占总耗时 76%）。

**第三轮优化 [K2, F2]**（pypardiso + 插值权重预计算）— `results/gpu_benchmark_2.json`：

| 步骤 | [A-H] CPU | [A-H]+pardiso+interp CPU | [A-H]+pardiso+interp GPU | 加速比 |
|------|-----------|-------------------------|-------------------------|--------|
| forward | 349ms | **214ms** | **183ms** | **1.6x** |
| reco | 32ms | **30ms** | **10ms** | 1.1x / 3.2x |
| interp | 73ms | **1.2ms** | **0.9ms** | **61x** |
| **total** | **458ms** | **250ms** | **198ms** | **1.8x / 2.3x** |

pypardiso（Intel PARDISO）替代 scipy SuperLU，forward 加速 1.6x；插值权重矩阵预计算使 interp 从 73ms 降至 1.2ms（61x）。

**第四轮优化 [K3]**（降秩精确直解 76→15 RHS）— `results/gpu_benchmark_8.json`：

| | 无 K3 (76 RHS) | 有 K3 (15 RHS) | 加速比 |
|--|--|--|--|
| CPU forward | 214ms | **86ms** | **2.5x** |
| CPU total | 250ms | **117ms** | **2.1x** |
| GPU forward | 183ms | **87ms** | **2.1x** |
| GPU total | 198ms | **95ms** | **2.1x** |

`Injref` 的 76 列激励仅张成 15 维子空间（数学证明见 `GUIDE/injref_rank15_proof.md`，实验验证见 `GUIDE/forward_batched_pcg_optimization.md`）。SVD 分解后只解 15 列 RHS 再线性重构，精度为机器精度（相对误差 6.8e-11）。**累计（pypardiso CPU）：原始 23210ms → 现在 117ms = ~198x 总加速。**

---

## 已完成的优化

### [A] A0 刚度矩阵组装：向量化 + COO + 几何预计算

**文件**：`src/ktc_methods/KTCFwd.py`

**原始实现**：Python for 循环遍历所有 HN 个单元，逐个调用 `grinprod_gauss_quad_node` 计算 6x6 局部刚度矩阵（3 点高斯积分，每次调用 `np.linalg.inv` 和 `np.linalg.det` 计算 2x2 Jacobian）。

**优化后**：分为两阶段——

**阶段一（`__init__` 预计算，仅执行一次）**：网格几何不随样本变化，一次性计算全部 HN 个单元的 Jacobian、逆矩阵、行列式、梯度矩阵和 G^T G 外积，缓存为 `_quad_GtG`（3 组 (HN, 6, 6) 数组）和 `_quad_abs_det`（3 组 (HN,) 数组）。同时预计算 COO 索引数组 `_row_A0`、`_col_A0`。

**阶段二（`SolveForward` 每次调用）**：仅做 sigma 加权累加和 COO 矩阵构建。

```python
# 每次调用仅需：
all_ss = sigma[self._H_vertex_idx]        # sigma 索引
for qq in range(3):
    sigma_w = all_ss @ self._quad_S[qq]   # (HN,) 加权
    all_Ke += (w * sigma_w * abs_det)[:, None, None] * GtG  # 缩放累加
A0 = coo_matrix((all_Ke.ravel(), (row, col))).tocsr()  # COO 一次性构建
```

| 操作 | 优化前 | 优化后 |
|------|--------|--------|
| 单元数据聚合 | `g[ind, :]` 逐个 | `g[H]` 一次性 → (HN, 6, 2) |
| Jacobian 计算 | `L @ g` 逐个 | `__init__` 预计算，`SolveForward` 直接使用缓存 |
| 2x2 矩阵求逆 | `np.linalg.inv` 逐个 | 解析公式，向量化，`__init__` 预计算 |
| G^T G 外积 | `einsum` 逐个 | `__init__` 预计算并缓存 |
| 矩阵构建 | 预分配数组 + CSR | COO `(data, (row, col))` → `tocsr()` |

**实测效果**：forward 总耗时从 1026ms 降至 334ms（**3.1x 加速**），其中组装部分加速远超 10x（剩余耗时主要是 `spsolve` 求解）。

---

### [B] S0 电极边界组装：COO 替代 CSR + 缓存

**文件**：`src/ktc_methods/KTCFwd.py`

**原始实现**：直接修改 CSR 稀疏矩阵 `M[i,j] += value` 和 `K[i,j] += value`，触发 `SparseEfficiencyWarning`（CSR 每次插入需复制整个矩阵内存）。

**优化后**：
1. **COO 收集**：循环中仅追加 (row, col, val) 到列表，循环结束后 `coo_matrix(...).tocsr()` 一次性构建，消除 `SparseEfficiencyWarning`
2. **S0 缓存**：S0 仅依赖 z（接触阻抗），而 z 在数据生成过程中恒定不变。首次计算后缓存为 `_S0_cached`，后续调用直接复用。通过 `np.array_equal` 检测 z 是否变化，确保正确性
3. `sp.sparse.diags(s.flatten())` 替代 `sp.sparse.csr_matrix(np.diag(s.flatten()))`，避免创建密集 (Nel x Nel) 中间矩阵

**实测效果**：第 2 个样本起 S0 组装耗时为 0（缓存命中），消除了 `SparseEfficiencyWarning`。

---

### [C] RHS 向量与 QC 矩阵预计算

**文件**：`src/ktc_methods/KTCFwd.py`

原始代码每次 `SolveForward` 调用都重新计算：
- `self.b = np.concatenate(zeros, C.T * Inj)` — 仅依赖 Inj 和 C（常量）
- `self.QC = np.block(zeros, Mpat.T @ C)` — 仅依赖 Mpat 和 C（常量）
- `self.Mpat.T * self.C * theta[ng2:, :]` — 其中 `Mpat.T * C` 是常量

**优化后**：全部移至 `__init__` 预计算为 `self.b`、`self.QC`、`self._MpatC`，`SolveForward` 直接使用。

---

### [D] reconstruct_list：对角矩阵算法优化

**文件**：`src/reconstruction/linearised_reco.py`

**原始实现**：构建完整 M x M 密集对角矩阵 `GammaInv = np.diag(gamma_vec)`，再计算 `BJ.T @ GammaInv @ BJ`（O(M^2) 内存，O(M^2 * N) 计算）。

**优化后**：向量化行缩放 `BJ_w = BJ * gamma_vec[:, None]`，再 `JGJ = BJ_w.T @ BJ`（O(M*N) 内存，O(M * N^2) 计算）。CPU 和 GPU 路径均受益。

```python
# 优化前（M=2356，浪费）：
GammaInv = np.diag(gamma_vec)           # (2356, 2356) 密集对角
JGJ = BJ.T @ GammaInv @ BJ             # O(M^2 * N)

# 优化后（向量化）：
BJ_w = BJ * gamma_vec[:, None]          # (M, N) 行缩放
JGJ = BJ_w.T @ BJ                      # O(M * N^2)
```

---

### [E] reconstruct_list：CuPy GPU 加速

**文件**：`src/reconstruction/linearised_reco.py`

`__init__` 中将 BJ、Rtv、Rsm 矩阵预加载到 GPU（`cp.asarray`）。每个样本的计算（对角缩放 + JGJ + 5 次 `cp.linalg.solve`）全部在 GPU 上完成，通过 `_reconstruct_list_gpu` 路径实现。

**实测效果**：reco 步骤 22801ms (CPU) → 3231ms (GPU) = **7.1x 加速**

---

### [F] interpolate_to_image：Delaunay 三角化缓存

**文件**：`src/reconstruction/linearised_reco.py`

**原始实现**：每次调用 `LinearNDInterpolator(self.pos, sigma)` 都重建 Delaunay 三角化，但网格质心 `self.pos` 恒定不变。

**优化后**：`__init__` 中预计算 `self._tri = Delaunay(self.pos)` 和像素网格 `self._pixcenters`，`interpolate_to_image` 直接复用缓存的三角化。

**实测效果**：interp 步骤 93ms → 63-69ms（~1.4x 加速，每次节省约 6ms，每个样本调用 5 次）。

---

### [G] SolveForward 求解策略：保持 CPU 稀疏求解

**文件**：`src/ktc_methods/KTCFwd.py`

最初尝试将 `scipy.sparse.linalg.spsolve` 替换为 `cp.linalg.solve(A_gpu, b_gpu)`（密集 GPU 求解），导致 **6.7x 性能倒退**（forward 从 1026ms 恶化到 6849ms）。

**根因分析**：
- `A.toarray()` 将稀疏矩阵转为密集矩阵（~288MB），丧失稀疏性
- O(N^3) 密集 LU 分解 vs O(N^1.5) 稀疏求解 — 计算量膨胀数百倍
- 每次调用需 CPU→GPU→CPU 传输 288MB 数据
- CuPy `spsolve` 不支持多列 RHS（76 列电流模式），无法直接使用稀疏 GPU 求解

**决策**：保持 CPU `scipy.sparse.linalg.spsolve`。配合上述组装优化，forward 总耗时从 1026ms 降至 334ms。

---

### [H] 预计算线性化重建投影矩阵

**文件**：`src/reconstruction/linearised_reco.py`

线性化重建中 `reconstruct_list` 每个样本需执行 5 次 `np.linalg.solve`（或 GPU 上的 `cp.linalg.solve`），耗时 3231ms（GPU）/ 22801ms（CPU），是整个流程的绝对瓶颈。

**关键洞察**：对于固定的 5 组 alpha 参数，求解矩阵 R_alpha = (J^T Gamma^-1 J + alpha*L)^-1 @ J^T @ Gamma^-1 是**完全静态的**。将其在 `__init__` 中一次性预计算后，每个样本的重建退化为 5 次矩阵-向量乘法。

**噪声模型选择**：预计算使用基于 Uref 的固定噪声模型（而非每样本 deltaU 自适应）。由于训练数据噪声实际由 `SetInvGamma(Uelref)` 生成，此模型与真实噪声分布更一致。

```python
# __init__ 中预计算 (仅执行一次):
gamma_fixed = 1.0 / ((noise_std1/100 * |Uref|)^2 + (noise_std2/100 * max(|Uref|))^2)
BJ_w = BJ * gamma_fixed[:, None]
JGJ = BJ_w.T @ BJ
for alpha in alphas:
    A = JGJ + alpha[0]*Rtv + alpha[1]*Rsm + alpha[2]*diag(JGJ)
    R = np.linalg.solve(A, BJ_w.T)   # (N_elements, N_measurements)
    self._R_precomputed.append(R)

# 每样本仅需:
delta_sigma = R @ deltaU              # 一次矩阵-向量乘
```

**内存优化**：使用 `A[diag_idx, diag_idx] += ...` 替代 `np.diag(jgj_diag)` 避免创建 (N, N) 对角矩阵，减少 ~360MB 峰值内存。

**实测效果**：reco 从 3231ms (GPU) / 22801ms (CPU) 降至 **32ms (CPU)**（**101x / 712x 加速**）。GPU 已无必要。

---

### [K3] SolveForward：降秩精确直解（76→15 RHS）

**文件**：`src/ktc_methods/KTCFwd.py`

**关键发现**：当前 `ref.mat` 中的 76 个电流激励 `Injref` 仅张成 15 维子空间（rank=15）。这是因为只有 16 个偶数编号电极参与注流且满足电流守恒（16-1=15）。完整数学证明见 `GUIDE/injref_rank15_proof.md`。

**优化原理**：对 RHS 矩阵 `self.b` 做 SVD 分解，提取 15 个基列 `B_basis` 和系数矩阵 `C`，使得 `B_full = B_basis @ C`。求解时只解 15 列，再通过矩阵乘法重构完整 76 列解。这是精确变换，不引入任何近似误差。

```python
# __init__ 中（仅执行一次）：
U, s, Vt = np.linalg.svd(self.b, full_matrices=False)
rank = int(np.sum(s > s[0] * 1e-12))  # = 15
self._b_basis = U[:, :rank] * s[:rank]   # (14899, 15)
self._b_coeff = Vt[:rank, :]              # (15, 76)

# SolveForward 中：
UU_basis = spsolve(self.A, self._b_basis)  # 解 15 列（原先 76 列）
UU = UU_basis @ self._b_coeff              # 线性重构回 76 列
```

**精度验证**：
- RHS 重构相对误差：7.8e-15（机器精度）
- 测量电压 Uel 与原方法完全一致

**实测效果**（40 样本 benchmark，pypardiso 环境）：
- CPU forward：214ms → **86ms** = **2.5x 加速**
- GPU forward：183ms → **87ms** = **2.1x 加速**
- CPU total：250ms → **117ms** = **2.1x 加速**

---

### [I] CPU 多进程流水线

**文件**：`scripts/generate_data.py`

通过 `--workers N` 参数启用多进程。使用 `ProcessPoolExecutor` 将样本生成拆分为多个块，每个 Worker 独立初始化 EITFEM + LinearisedRecoFenics 后并行处理各自的样本子集。

```bash
python scripts/generate_data.py --level 1 --num-images 2000 --workers 4
```

**限制**：仅支持 CPU 模式（GPU 上下文不跨进程共享）；HDF5 输出不兼容多进程。

**预期收益**：4+ 核 CPU 可提升 2-3x 吞吐量（每 Worker 独立进行 forward 求解）。

---

### [J] HDF5 批量存储

**文件**：`scripts/generate_data.py`

通过 `--hdf5` 参数启用。将逐样本 `.npy` 文件替换为单个 HDF5 文件 `data.h5`：

```bash
python scripts/generate_data.py --level 1 --num-images 2000 --hdf5
```

HDF5 文件包含三个 Dataset：`gt`(N,256,256)、`measurements`(N,M)、`reco`(N,5,256,256)。

**收益**：消除海量小文件的文件系统开销，PyTorch DataLoader 可通过切片读取加速训练。

---

### [K2] SolveForward 求解器：pypardiso 替代 scipy SuperLU

**文件**：`src/ktc_methods/KTCFwd.py`

`scipy.sparse.linalg.spsolve` 底层使用 SuperLU 求解器。Intel PARDISO（通过 `pypardiso` 包）针对稀疏 SPD 矩阵有深度优化，支持多线程 LU 分解和多列 RHS。

```python
try:
    from pypardiso import spsolve as pardiso_spsolve
    _HAS_PARDISO = True
except ImportError:
    _HAS_PARDISO = False

# SolveForward 中:
if _HAS_PARDISO:
    UU = pardiso_spsolve(self.A, self.b)
else:
    UU = sp.sparse.linalg.spsolve(self.A, self.b)
```

自动检测：安装 `pypardiso` 后自动启用，未安装则回退到 scipy。

**实测效果**：forward 从 349ms 降至 **214ms**（**1.6x 加速**）。

---

### [F2] interpolate_to_image：稀疏权重矩阵预计算

**文件**：`src/reconstruction/linearised_reco.py`

**原始实现**（优化 [F] 后）：每次调用 `LinearNDInterpolator(self._tri, sigma)` 仍需遍历所有像素点计算重心坐标，耗时 ~14ms/次 × 5 = ~73ms/样本。

**优化后**：`__init__` 中利用 Delaunay 三角化的 `transform` 矩阵，一次性计算所有像素的重心坐标并构建稀疏插值权重矩阵 W (n_pixels × n_elements)。`interpolate_to_image` 退化为 `W @ sigma` 一次稀疏矩阵-向量乘法。

```python
# __init__ 中预计算:
simplex_idx = self._tri.find_simplex(self._pixcenters)
# ... 计算重心坐标 → 构建 CSR 稀疏矩阵 W

# 每次调用仅需:
def interpolate_to_image(self, sigma):
    return np.flipud((self._interp_W @ sigma.flatten()).reshape(256, 256))
```

**数值验证**：与原始 `LinearNDInterpolator` 输出的最大误差为 2.2e-16（机器精度）。

**实测效果**：interp 从 73ms 降至 **1.2ms**（**61x 加速**）。

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/ktc_methods/KTCFwd.py` | [A] 向量化 A0 组装、[B] COO S0 组装、[C] 预计算缓存、[K2] pypardiso、[K3] 降秩直解 |
| `src/reconstruction/linearised_reco.py` | [D] 对角优化、[E] CuPy GPU、[F] Delaunay 缓存、[H] 投影矩阵预计算、[F2] 插值权重矩阵 |
| `scripts/generate_data.py` | `--gpu`/`--hdf5`/`--workers` 参数、[I] 多进程、[J] HDF5 |
| `scripts/benchmark_data_gen.py` | CPU vs GPU 基准测试（自动编号、优化项追踪） |

---

## 潜在的未来优化方向

### 高收益

**K. GPU 预处理共轭梯度法（CG）替代 CPU 直接求解**（已被 [K3] 降秩直解取代）

经 `GUIDE/forward_batched_pcg_optimization.md` 中的实验验证，GPU Jacobi-PCG 在当前环境下无法同时达到高精度和高速度。降秩精确直解 [K3] 以更简单的方式获得了更大的加速（3.7x），且零误差、零 GPU 依赖。

---

**L. CPU-GPU 异步流水线**

不同于多进程方案（B），此方案使用 Python 多线程 + CUDA Streams 实现 CPU 与 GPU 的重叠执行：

```
线程 1 (CPU): [phantom_N+1 生成 + 组装]  [phantom_N+2 生成 + 组装]  ...
线程 2 (GPU): [样本 N 的 solve + reco]    [样本 N+1 的 solve + reco]  ...
```

由于 CPU 组装矩阵和 GPU 矩阵运算互不干扰（GIL 在 CuPy 调用期间释放），可将 phantom 生成和 FEM 组装的耗时完全隐藏在 GPU 计算背后。实现比多进程简单，无需进程间通信。

### 中等收益

**M. 稀疏 Cholesky 分解**

EIT 刚度矩阵 A 是对称正定（SPD）矩阵，使用 Cholesky 分解可比通用 LU 分解节省一半计算量：

```python
# 需要 scikit-sparse（仅 Linux/Mac）
from sksparse.cholmod import cholesky
factor = cholesky(self.A)
UU = factor(self.b)
```

预期：求解步骤从 ~100ms 降至 ~50ms。

---

**N. Numba JIT 加速 S0 电极边界组装**

S0 电极边界循环仍是 Python for-loop（含 `np.where` 查找）。`@numba.njit` 可将其编译为机器码。但由于 S0 已被缓存（仅首次调用计算），实际收益仅体现在 EITFEM 初始化阶段。

### 低收益（超大规模场景）

**O. S0 电极边界组装全向量化**

将电极边界循环完全向量化——预计算每个电极对应的单元，批量调用 `bound_quad1` 和 `bound_quad2`。实现复杂度较高（每个电极的单元数不固定），且 S0 已被缓存，收益有限。

---

**P. 混合精度（float32）**

FEM 组装和求解使用 float32 代替 float64，内存带宽减半，SIMD 吞吐翻倍。需要仔细验证重建质量是否受影响。

---

**Q. CUDA 自定义核函数 + Colored pJDS 稀疏格式**

通过 CuPy `RawKernel` 或 Numba CUDA 编写 FEM 组装的 CUDA 核函数，彻底消除 Python 开销。

进阶方案：传统 CSR 格式在 GPU 并行组装时会产生写入冲突。可采用 **Colored pJDS（着色填充锯齿对角线存储）**格式——预先对网格节点运行图着色算法，确保相邻节点分配不同颜色，使 GPU 线程可安全并行写入。将着色索引数组传给 CuPy `RawKernel`。

实现成本高，但可将组装时间从 ~300ms 压缩至 1-2ms。

**学术依据**：EIT GPU 加速论文中提出的高效稀疏存储方案，专为 FEM 并行组装和三角求解设计。
