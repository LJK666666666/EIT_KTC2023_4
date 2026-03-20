# Forward Batched PCG Optimization Notes

## 1. Goal

本文档说明对 `scripts/generate_data.py` 中 forward 步骤所做的一轮独立优化试验。目标不是直接改原脚本，而是先回答三个问题：

1. `K. GPU 预处理共轭梯度法（CG）替代 CPU 直接求解` 这条思路在当前代码库里应当如何落地。
2. 在当前机器和当前矩阵规模下，GPU batched PCG 是否真的比现有 forward 更快。
3. 如果 GPU PCG 不是最优，当前 forward 系统里还有没有更直接、更稳妥的加速点。

本轮试验的实现脚本是：

- `scripts/benchmark_forward_batched_pcg.py`

基准结果保存在：

- `results/forward_batched_pcg_l1_n5_1/summary.json`
- `results/forward_batched_pcg_l1_n5_1/forward_time_comparison.png`
- `results/forward_batched_pcg_l1_n5_1/forward_pareto.png`


## 2. Baseline Forward Structure

当前 forward 由 `src/ktc_methods/KTCFwd.py` 中的 `EITFEM.SolveForward()` 完成，其核心可以拆成三部分：

1. 组装 `A0`
2. 组装或复用边界项 `S0`
3. 求解线性系统

系统形式是：

```text
A * Theta = B
```

其中：

- `A` 为每个样本对应的稀疏对称矩阵，尺寸约 `14899 x 14899`
- `B` 为固定右端项，尺寸 `14899 x 76`
- `Theta` 为全部 76 个电流模式对应的解

原实现的关键瓶颈在第 3 步：

- 如果安装了 `pypardiso`，走 `pardiso_spsolve(A, B)`
- 否则走 `scipy.sparse.linalg.spsolve(A, B)`

这意味着每个样本都要对一个大规模稀疏系统做一次直接求解。


## 3. 为什么会想到 GPU Batched PCG

`GUIDE/data_generation_optimization.md` 的 K 节给出的判断是对的：

- `A` 是对称系统
- 稀疏直接解在 CPU 上通常是主要瓶颈
- 如果能在 GPU 上做稀疏迭代法，理论上可以减少 factorization 成本

但“直接把 76 个 RHS 逐列丢给 GPU CG”并不是当前问题最好的表述，因为这里还有一个更重要的结构性质。


## 4. 关键发现：76 个 RHS 实际秩只有 15

在试验脚本里，我先对 `B` 做了列空间分析，发现：

```text
B.shape = (14899, 76)
rank(B) = 15
```

这意味着 76 个 RHS 并不线性独立，可以写成：

```text
B_full = B_basis * C
```

其中：

- `B_basis` 只需要 15 列
- `C` 为 `15 x 76` 的重构系数矩阵

于是原问题可以改写成：

```text
A * Theta_basis = B_basis
Theta_full = Theta_basis * C
```

这个变换有两个直接收益：

1. 无论用直接解还是迭代解，真正需要求解的 RHS 数量都从 76 下降到 15。
2. GPU batched PCG 不再需要维护 76 组搜索方向，规模明显更合理。

这是本轮试验里最重要的结构性优化点。


## 5. 试验路线

本轮实际比较了 4 类方案。

### 5.1 Current CPU Direct

这条路径等价于当前 `SolveForward()`：

- 组装 `A`
- 直接求解 `A * Theta = B_full`
- 提取测量向量 `Uel`

它作为误差基准，记为 `current_cpu_direct`。

### 5.2 CPU Reduced Direct Exact

这是一个意外发现但非常有效的精确优化：

- 先将 `B_full` 降到 `B_basis`
- 只解 15 个 RHS
- 再用系数矩阵重构回 76 个解

这个方案是严格等价的精确解，不引入额外近似。

### 5.3 GPU Batched Jacobi-PCG

这一条才是对 K 节思路的真正实现。

实现形式不是 CuPy，而是 PyTorch CUDA sparse CSR。原因很简单：

- 本机 `cupyx.scipy.sparse.linalg` 缺 CUDA 相关 DLL
- 直接用 CuPy 稀疏线代跑不起来
- 但 PyTorch 的 CUDA sparse CSR 可用

所以本轮 GPU PCG 的底层实现为：

- 稀疏矩阵：`torch.sparse_csr_tensor`
- 稀疏乘密：`torch.sparse.mm`
- 预条件器：Jacobi，对角倒数
- 批处理方式：15 列 RHS 同时做 batched PCG

### 5.4 Background Warm Start

GPU PCG 如果从零初值开始，迭代数偏大。因此我额外加入了背景导电率的 warm start：

1. 先对背景 conductivity 组装 `A_bg`
2. 精确解出背景基解 `Theta_bg_basis`
3. 对每个新样本，将 `Theta_bg_basis` 作为 PCG 初值

这样做的直觉是：

- 大多数样本的背景区域与 `A_bg` 相近
- inclusion 只改变局部 conductivity
- 背景解可以作为一个合理的初始近似


## 6. GPU Batched PCG 的具体算法

这里实现的不是 block-CG，而是“多 RHS 同步推进的逐列 CG”。

原因是：

- 标准 block-CG 需要求解 `P^T A P`
- 当 RHS 存在线性相关时，这个小矩阵容易奇异或数值不稳定
- 当前问题原始 RHS 就是高度相关的

因此这里采用更稳的 batched 版本：

```text
for k in range(maxiter):
    AP = A @ P
    alpha_j = (r_j^T z_j) / (p_j^T A p_j)    for each RHS j
    x_j = x_j + alpha_j * p_j
    r_j = r_j - alpha_j * A p_j
    z_j = M^{-1} r_j
    beta_j = (r_j^T z_j) / (r_j_old^T z_j_old)
    p_j = z_j + beta_j * p_j
```

区别只在于：

- `A @ P` 是一次稀疏乘密矩阵乘法
- `P`、`R`、`Z` 都是 `N x 15` 的 dense matrix
- 每一列的 `alpha_j`、`beta_j` 分别独立计算

这样就兼顾了：

- GPU 上一次处理多列 RHS
- 算法稳定性接近单列 CG


## 7. 为什么先做共享组装，再比较求解器

本轮试验不是简单地拿原 `SolveForward()` 和新方法做黑盒对比，而是把流程拆开：

1. 用与原实现完全一致的逻辑组装 `A`
2. 在同一个 `A` 上比较不同求解器
3. 都统一回到相同的 `Uel` 提取逻辑

这么做有两个好处：

1. 排除了 phantom、插值、噪声等非 forward 误差
2. 可以确保所有方法比较的是同一线性系统

我还额外验证了：

- `assemble_system + direct_solve + measurements_from_theta`
- 与原 `EITFEM.SolveForward()`

输出完全一致。


## 8. 实测结果

以 `results/forward_batched_pcg_l1_n5_1/summary.json` 为准，5 个随机样本的平均结果如下：

| Mode | Mean Forward Time (ms) | Mean Relative Error | Speedup vs Current |
|---|---:|---:|---:|
| `current_cpu_direct` | 374.2 | 0 | 1.00x |
| `cpu_direct_reduced_exact` | 74.3 | 1.61e-11 | 5.03x |
| `gpu_batched_pcg_iter200` | 264.3 | 5.31e-2 | 1.42x |
| `gpu_batched_pcg_iter600` | 410.5 | 1.65e-2 | 0.91x |
| `gpu_batched_pcg_iter1000` | 563.9 | 1.18e-3 | 0.66x |

这些数字说明了三件事。

### 8.1 GPU PCG 低迭代数确实能变快

`iter200` 时，GPU batched PCG 已经比当前实现快：

- `374.2 ms -> 264.3 ms`

但误差也比较大：

- `5.31%`

这个量级已经接近甚至进入 synthetic noise 的量级，不适合直接替代精确 forward。

### 8.2 GPU PCG 提高精度后会失去速度优势

当迭代数提升到：

- `600`：误差约 `1.65%`
- `1000`：误差约 `0.118%`

速度分别变成：

- `410.5 ms`
- `563.9 ms`

也就是说，在本机环境下，GPU Jacobi-PCG 没有形成一个比当前直接解更好的精度-速度平衡点。

### 8.3 真正最强的方案其实是 Reduced Exact Direct Solve

`cpu_direct_reduced_exact` 的结果非常干净：

- 完全精确
- 只解 15 个 RHS
- 速度约 `74.3 ms`
- 相对当前实现约 `5.03x`

它既没有迭代误差，也没有 GPU 依赖问题。


## 9. 为什么 GPU PCG 没赢

这是本轮试验里最值得说明的部分。不是 GPU PCG 思路错了，而是当前环境里它遇到了两个现实约束。

### 9.1 当前 baseline 太强

本机安装了 `pypardiso`，因此 baseline 不是普通 `scipy.spsolve`，而是更强的 PARDISO 稀疏直接解。

这会显著抬高“需要击败的 CPU baseline”。

换句话说：

- 如果 baseline 是普通 `scipy.spsolve`
- GPU PCG 可能会更有竞争力

但当前不是这个情况。

### 9.2 Jacobi 预条件器太弱

GPU PCG 目前只用了最简单的 Jacobi 预条件器，也就是：

```text
M^{-1} = diag(A)^{-1}
```

它的优点是简单、便宜、全 GPU 友好。

它的缺点也很明显：

- 对当前系统的谱性质改善有限
- 要达到较低误差，仍需要很多迭代

本轮我还做过额外试验，发现如果使用“背景矩阵的精确求解器”做预条件器，PCG 迭代数能从上千步降到十几步。但这个预条件器本身又太贵，整体仍不划算。


## 10. 当前最合理的结论

如果目标是“尽快把 `generate_data.py` 的 forward 真正加速起来”，当前最值得优先落地的方案不是 GPU PCG，而是：

```text
76 RHS -> rank-15 RHS -> exact direct solve -> reconstruct to 76 RHS
```

原因很直接：

1. 它是精确方法，不改变结果。
2. 它已经在当前机器上给出约 `5x` 的真实加速。
3. 它不依赖额外 GPU 稀疏线代栈，工程风险低。

如果目标是“继续沿着 K 节路线深入 GPU 求解器”，那下一步应当考虑的不是再微调 Jacobi-PCG，而是：

1. 更强的 GPU 预条件器
2. 真正可用的 CuPy / cuSPARSE / cuSOLVER 环境
3. 是否存在更适合该系统的 GPU sparse direct 或 mixed strategy


## 11. 建议的后续顺序

推荐按下面顺序推进。

### 路线 A：先拿到稳定收益

1. 基于当前实验结果，先写一个新的 `generate_data` 变体脚本。
2. 将 forward 从 76 RHS 直接解改为 15 RHS 精确解。
3. 做端到端吞吐测试，确认整体数据生成速度提升。

这条路线最稳，且结果不变。

### 路线 B：继续研究 GPU PCG

1. 保留 `scripts/benchmark_forward_batched_pcg.py` 作为试验平台。
2. 在不改原脚本的前提下继续试更强预条件器。
3. 如果未来 CuPy 稀疏线代环境恢复正常，再对比 PyTorch 版和 CuPy 版。


## 12. 复现实验

复现当前文档中的主要实验，直接运行：

```bash
python scripts/benchmark_forward_batched_pcg.py --num-samples 5 --pcg-iters 200 600 1000
```

如果只想看某一个 GPU 迭代档位：

```bash
python scripts/benchmark_forward_batched_pcg.py --num-samples 5 --pcg-iters 600
```

输出会自动保存到：

```text
results/forward_batched_pcg_l{level}_n{num_samples}_{num}/
```


## 13. Final Takeaway

这轮试验最重要的结论不是“GPU PCG 不行”，而是：

1. 这个问题的 RHS 有强结构性冗余，先降秩比直接上 GPU 更重要。
2. GPU Jacobi-PCG 在当前环境下已经跑通，但还没有达到最优性价比。
3. 当前 forward 的最佳可落地优化，其实是一个精确的 reduced-RHS direct solve。

因此，如果下一步只允许做一项改动，优先级应当是：

```text
Reduced-RHS exact solve > GPU Jacobi-PCG
```
