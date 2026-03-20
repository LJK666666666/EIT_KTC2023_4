# Why the 76 `Injref` Excitation Modes Reduce Exactly to 15 Basis Modes

## 1. Statement

需要先澄清一句话：

> 严格来说，不是“76 种电压测量模式等价于 15 种”，而是“当前 `ref.mat` 中的 76 个电流激励模式只张成一个 15 维子空间，因此对应的 PDE 解、电极电压和最终测量电压都可以由 15 个基激励的响应精确线性重构”。

本文给出完整证明。


## 2. Problem Setup

在本项目里，forward 求解使用完整电极模型（Complete Electrode Model, CEM）。对固定的导电率分布 `σ` 和接触阻抗 `z`，未知量包括：

- 域内电势 `u(x)`
- 电极电压向量 `U = (U_1, ..., U_L)^T`

输入是电极电流向量：

```text
I = (I_1, ..., I_L)^T
```

这里 `L = 32`。

满足的 PDE/边界条件写成：

```text
∇·(σ∇u) = 0                                      in Ω

u + z_l σ ∂u/∂n = U_l                            on E_l,   l = 1,...,L

∫_{E_l} σ ∂u/∂n ds = I_l                         for l = 1,...,L

σ ∂u/∂n = 0                                      on ∂Ω \ (∪_l E_l)

Σ_{l=1}^L I_l = 0
```

最后一个条件是电流守恒。


## 3. First Step: The Forward Map Is Linear in the Injected Current

### 3.1 Continuous PDE Level

固定 `σ` 和 `z` 后，CEM 关于 `(u, U)` 与输入 `I` 的关系是线性的。

更明确地说，若

```text
I = a I^(1) + b I^(2)
```

且 `(u^(1), U^(1))`、`(u^(2), U^(2))` 分别对应 `I^(1)`、`I^(2)` 的解，则

```text
u = a u^(1) + b u^(2)
U = a U^(1) + b U^(2)
```

对应输入 `I` 的解。

证明非常直接：

1. 对域内方程：

```text
∇·(σ∇u)
= ∇·(σ∇(a u^(1) + b u^(2)))
= a ∇·(σ∇u^(1)) + b ∇·(σ∇u^(2))
= 0
```

2. 对电极 Robin 条件：

```text
u + z_l σ ∂u/∂n
= a(u^(1) + z_l σ ∂u^(1)/∂n) + b(u^(2) + z_l σ ∂u^(2)/∂n)
= a U_l^(1) + b U_l^(2)
```

3. 对电极净电流条件：

```text
∫_{E_l} σ ∂u/∂n ds
= a ∫_{E_l} σ ∂u^(1)/∂n ds + b ∫_{E_l} σ ∂u^(2)/∂n ds
= a I_l^(1) + b I_l^(2)
```

因此 CEM 定义了一个线性算子：

```text
F_{σ,z} : I -> (u, U)
```

其中对我们最关心的是电极电压部分：

```text
G_{σ,z} : I -> U
```

也是线性的。


### 3.2 Measured Voltages Are Also Linear

在实际代码中，并不是直接把全部 `U` 当输出，而是再经过测量矩阵得到可见测量电压：

```text
y = M U
```

其中 `M` 是固定线性算子，所以

```text
H_{σ,z} : I -> y
```

同样是线性的。

于是只要输入电流激励 `I` 可线性表示，输出测量 `y` 也必然可线性表示。


## 4. Second Step: The Discrete Solver in This Repo Preserves This Linearity

代码里这一结构非常明确。

在 [`src/ktc_methods/KTCFwd.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/ktc_methods/KTCFwd.py#L96) 中，右端项写成：

```python
self.b = np.concatenate(
    (np.zeros((self.ng2, Inj.shape[1])), self.C.T * Inj), axis=0)
```

也就是说，离散系统是：

```text
A Θ = B
```

其中：

- `A` 只由 `σ` 和 `z` 决定
- `B` 由 `Inj` 线性决定

随后在 [`src/ktc_methods/KTCFwd.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/ktc_methods/KTCFwd.py#L195) 到 [`src/ktc_methods/KTCFwd.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/ktc_methods/KTCFwd.py#L199) 中：

```python
self.theta = UU
self.Pot = UU[0:self.ng2, :]
self.Umeas = self._MpatC * self.theta[self.ng2:, :]
self.Umeas = self.Umeas.T[self.mincl.T].T
```

所以从 `Inj` 到 `theta`，再到 `Umeas`，全程都是线性映射。

因此只要能证明 `Injref` 的 76 列只张成 15 维子空间，就立刻推出：

- 76 组离散 PDE 解只需 15 组基解
- 76 组电极电压只需 15 组基电压
- 76 组最终测量值只需 15 组基测量值


## 5. Third Step: Analyze the Actual `Injref` Used Here

现在进入关键部分：为什么当前 `ref.mat` 里的 76 个激励只有 15 个独立自由度。

我们实际检查 `KTC2023/Codes_Python/TrainingData/ref.mat` 中的 `Injref`，得到：

```text
shape(Injref) = (32, 76)
rank(Injref) = 15
```

更重要的是它的结构：

1. 所有奇数编号电极行全为 `0`
2. 每一列都只有两个非零元
3. 每一列都可以写成

```text
α (e_{2a} - e_{2b})
```

其中：

- `e_k` 是第 `k` 个电极对应的标准基向量
- `2a, 2b` 都是偶数编号电极
- `α` 是该列的注流幅值

例如前 16 列就包含：

```text
col 0  :  +1.472 e_0   -1.472 e_2
col 1  :  +1.4952 e_2  -1.4952 e_4
col 2  :  +1.4502 e_4  -1.4502 e_6
...
col 14 :  +1.454  e_28 -1.454  e_30
col 15 :  -1.5052 e_0  +1.5052 e_30
```

也就是说，当前激励实际上只作用在这 16 个偶数电极上：

```text
E_active = {0, 2, 4, ..., 30}
```


## 6. Dimension Upper Bound: Why It Cannot Exceed 15

定义子空间：

```text
W = { I ∈ R^32 : I_{odd} = 0,  Σ_{l=1}^{32} I_l = 0 }
```

由于：

1. 只有 16 个偶数电极位置允许非零
2. 总电流必须守恒

所以：

```text
dim(W) = 16 - 1 = 15
```

而 `Injref` 的每一列都在 `W` 中，因此

```text
rank(Injref) <= 15
```

这给出了上界。


## 7. Dimension Lower Bound: Why It Is At Least 15

现在要证明不是更小，而是刚好等于 15。

观察 `Injref` 的前 15 列，它们正好对应 16 个活跃电极上的相邻差分：

```text
g_1  = e_0  - e_2
g_2  = e_2  - e_4
g_3  = e_4  - e_6
...
g_15 = e_28 - e_30
```

`Injref` 的前 15 列只是这些向量分别乘上非零标量，因此它们与 `g_1, ..., g_15` 具有相同的线性独立性。

下面证明 `g_1, ..., g_15` 线性独立。

设

```text
Σ_{k=1}^{15} c_k g_k = 0
```

展开得：

```text
c_1 e_0
(-c_1 + c_2) e_2
(-c_2 + c_3) e_4
...
(-c_14 + c_15) e_28
(-c_15) e_30
= 0
```

由于 `e_0, e_2, ..., e_30` 线性独立，所以各系数必须逐项为零：

```text
c_1 = 0
-c_1 + c_2 = 0  => c_2 = 0
-c_2 + c_3 = 0  => c_3 = 0
...
-c_15 = 0       => c_15 = 0
```

于是

```text
c_1 = c_2 = ... = c_15 = 0
```

这说明 `g_1, ..., g_15` 线性独立。

因此

```text
rank(Injref) >= 15
```

结合前面的上界，

```text
rank(Injref) = 15
```


## 8. Exact Representation of All 76 Excitations by 15 Basis Excitations

因为 `rank(Injref) = 15`，存在 15 个基激励列组成矩阵 `I_basis`，以及系数矩阵 `C`，使得

```text
Injref = I_basis C
```

这不是近似，而是精确分解。

于是对于固定 `σ, z` 的 forward map `H_{σ,z}`，有：

```text
Y_full = H_{σ,z}(Injref)
       = H_{σ,z}(I_basis C)
       = H_{σ,z}(I_basis) C
```

这里最后一步用到的就是线性性。

换成逐列表述更直观：

若第 `k` 个激励满足

```text
I^(k) = Σ_{j=1}^{15} c_{jk} I_basis^(j)
```

那么对应的：

- 域内解 `u^(k)`
- 电极电压 `U^(k)`
- 测量电压 `y^(k)`

都满足

```text
u^(k) = Σ_{j=1}^{15} c_{jk} u_basis^(j)
U^(k) = Σ_{j=1}^{15} c_{jk} U_basis^(j)
y^(k) = Σ_{j=1}^{15} c_{jk} y_basis^(j)
```

因此确实**不需要独立求解 76 次**。


## 9. What This Means for the Forward Solver

原始离散系统是：

```text
A Θ_full = B_full
```

其中 `B_full` 的 76 列来自 `Injref`。

由于 `rank(B_full) = rank(Injref) = 15`，我们可以取一个 15 列基：

```text
B_full = B_basis C
```

只需先求：

```text
A Θ_basis = B_basis
```

再重构：

```text
Θ_full = Θ_basis C
```

然后测量输出自动满足：

```text
Y_full = M Θ_full = M Θ_basis C
```

这就是前面优化实验里“76 RHS -> 15 RHS reduced exact solve”的数学依据。


## 10. Important Clarification

这里的结论是**针对当前这份 `Injref`** 的，不是对任意 32 电极 EIT 都成立。

如果允许 32 个电极都参与注流，并满足总电流守恒，那么一般可用激励空间维数是：

```text
32 - 1 = 31
```

当前之所以只有 15 维，是因为：

1. 只使用了 16 个偶数编号电极作为活跃注流电极
2. 同时还满足总电流和为零

所以才变成：

```text
16 - 1 = 15
```

因此：

- `15` 不是 PDE 本身强加的
- `15` 是这份 `ref.mat` 的激励设计决定的


## 11. Final Conclusion

现在可以把完整结论写成一个严格命题。

### Proposition

对固定的导电率分布 `σ` 和接触阻抗 `z`，设 `Injref ∈ R^{32×76}` 为当前 `ref.mat` 中的电流激励矩阵。若：

1. 每一列都只在 16 个偶数电极上非零
2. 每一列满足总电流守恒
3. 前 15 个相邻偶数电极差分激励线性独立

则：

```text
rank(Injref) = 15
```

并且由于 CEM forward map 关于输入电流是线性的，76 个激励对应的：

- PDE 解
- 电极电压
- 测量电压

都可以由 15 个基激励对应的响应精确线性重构。

### Therefore

在当前项目里，可以严格地把：

```text
76 次 RHS 求解
```

改写为：

```text
15 次基 RHS 求解 + 线性重构
```

且不引入任何近似误差。
