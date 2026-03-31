### 痛点二：`SolveForward` (正问题有限元求解) 的 CPU 瓶颈

循环中有这样一行代码：`Uel_sim = solver.SolveForward(sigma_gt, z)`。注：`Uel_sim = np.asarray(solver.SolveForward(sigma_gt, z)).reshape(-1, 1)`。
它调用了 `src.EITFEM` 类。在这个类的内部，肯定隐藏着使用 `scipy.sparse.linalg.spsolve` 或类似方法求解大型矩阵方程 $Ax = b$ 的代码。

**💡 解决方案：使用 CuPy 进行平滑的 CUDA 替换**
你需要打开 `src/EITFEM.py` 文件（或者包含该类的文件），找到 `SolveForward` 函数中解方程的那一行。

**假设原代码长这样 (非常典型的 SciPy 写法)：**

```python
import scipy.sparse.linalg as sla

# A 是稀疏刚度矩阵，b 是电流向量
V = sla.spsolve(A, b) 

```

**你只需要改成这样：**

```python
import cupyx.scipy.sparse.linalg as csla
import cupyx.scipy.sparse as csparse
import cupy as cp

# 将 CPU 稀疏矩阵搬到 GPU
A_gpu = csparse.csr_matrix(A) 
b_gpu = cp.asarray(b)

# 在 GPU 上求解
V_gpu = csla.spsolve(A_gpu, b_gpu)

# 将结果搬回 CPU 继续后续计算
V = cp.asnumpy(V_gpu)

```

仅仅这几行改动，对于大型网格，通常能带来 **5到10倍** 的提速。

---

### 痛点三：`reconstruct_list` (基于 FEniCS 的线性重建) 的巨量运算

循环中还有极其致命的一步：`delta_sigma_list = reconstructor.reconstruct_list(Uel_noisy, alphas)`。
你不仅在生成正问题数据，同时还在为每个样本计算 5 种传统算法（TV, LM, Smoothness 等）的重建图像，以此作为神经网络的输入（这是一种典型的 Physics-Informed 预处理策略）。

由于这里使用了 FEniCS (一个主要跑在 CPU 上的有限元框架)，这部分极大概率是整个循环里**最慢的**（你可以看看控制台打印的 `Reconstruction: ... s` 时间是不是比 `Simulate Measurements` 还要长得多）。

**💡 解决方案：预计算雅可比矩阵并迁移至 GPU**
因为这里是“线性化重建 (Linearised Reco)”，它的数学本质通常是：


$$\Delta \sigma = (J^T J + \alpha R)^{-1} J^T \Delta U$$


在固定的水箱网格下，雅可比矩阵 $J$ 和正则化矩阵 $R$ 是**恒定不变**的！如果 `LinearisedRecoFenics` 内部每次都在重复计算这些矩阵的求逆，那简直是灾难。

**优化思路：**

1. 检查 `LinearisedRecoFenics` 的内部代码，看它是否预先计算并缓存了算子 $H = (J^T J + \alpha R)^{-1} J^T$。
2. 如果算子 $H$ 已经缓存为 NumPy 稠密矩阵，你可以直接用 PyTorch 或 CuPy 将 $H$ 放到 GPU 上，每次循环只需要做一次简单的矩阵向量乘法运算：
`delta_sigma_gpu = H_gpu @ delta_U_gpu`
这能把原本需要好几秒的重建时间，压缩到 **零点几毫秒**！
