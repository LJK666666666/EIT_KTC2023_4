## Harmonic Cascaded DPCA-UNet (HC-DPCA-UNet)

### 🌐 架构全景数据流 (Global Data Flow)

$$\text{Voltage} \, (1D) \xrightarrow{\text{Module 1}} K, V \quad \big|\quad \text{Pixels} \, (2D) \xrightarrow{\text{Module 2}} Q$$
$$Q, K, V \xrightarrow{\text{Module 3 (Cascaded)}} \text{Feature Map}_{128} \xrightarrow{\text{Module 4 (UNet)}} \text{Segmentation}_{3}$$

---

### 🧱 模块 1：谐波电极编码器 (Harmonic Electrode Encoder)
**使命：提取全局电场畸变规律，生成带空间物理锚点的 $K$ 和 $V$。**

1.  **输入拆分：**
    * 纯电压特征：形状为 `(B, 31, 76)`，代表 31 个差分通道在 76 种激励下的响应。
    * 电极极坐标 $\theta$：形状为 `(31,)`。
2.  **逼近极限的高频谐波注入 (NeRF-style L=8)：**
    * 榨干 32 个电极的物理分辨率上限，对角度 $\theta$ 展开 **8 阶**傅里叶谐波：$[\sin(\theta), \cos(\theta), \sin(2\theta), \cos(2\theta), \dots, \sin(8\theta), \cos(8\theta)]$。
    * 得到极致锐利的纯坐标特征：形状为 `(B, 31, 16)`。
3.  **非线性双轨升维与无损相加（防淹没 + 流形折叠）：**
    * **电压轨 (语义直通车)：** 直接经过 `Linear(76, 128)` $\rightarrow$ `Voltage_128d`。
    * **坐标轨 (非线性折叠)：** 经过带有激活函数的 MLP：`Linear(16, 64) -> GELU -> Linear(64, 128)` $\rightarrow$ `Coord_128d`。这一步打破了纯数学的死板，让坐标真正融入高维语义隐空间。
    * **等权融合：** $X = \text{Voltage\_128d} + \text{Coord\_128d}$。此时 $X$ 形状为 `(B, 31, 128)`，空间水印与电信号完美契合。
4.  **电极自注意力层 (Self-Attention)：**
    * 输入 $X$ 经过 1 层标准的 `TransformerEncoderLayer`（包含 Multi-Head Self-Attention + LayerNorm + FFN + 残差连接）。
    * 这一步让 31 个电极互相“对账”和“通讯”，提取全局电场的非局部协方差特征（例如远端电极的联动畸变）。
5.  **生成键值对：**
    * 融合后的特征分别经过两个独立的 `Linear(128, 128)`，得到最终的 $K$ 和 $V$，形状均为 `(B, 31, 128)`。

---

### 🧱 模块 2：谐波空间查询器 (Harmonic Spatial Query Generator)
**使命：为图像空间的 $256 \times 256$ 个像素点生成高分辨率、包含全频谱物理特征的查询向量 $Q$。**

1.  **基础物理坐标：**
    * 生成 $256 \times 256$ 个网格点，提取最基础的低频空间特征：横坐标 $x$、纵坐标 $y$、以及距离水箱中心的距离 $r$。均为以水箱半径为1进行归一化后的结果。
2.  **全频谱空间谐波注入：**
    * 对 $x$ 施加 8 阶谐波：$[\sin(x), \cos(x), \dots, \sin(8x), \cos(8x)]$（共 16 维）。
    * 对 $y$ 施加 8 阶谐波：$[\sin(y), \cos(y), \dots, \sin(8y), \cos(8y)]$（共 16 维）。
    * **统一拼接 (Concat)：** 将低频基础特征 $(x, y, r)$ 与所有高频谐波强行拼接。形成一个极其丰满的“全频谱空间向量”，总长度为 $3 + 16 + 16 = 35$ 维。
3.  **深层统一 MLP 升维 (特征深度交互)：**
    * 将 35 维的全频谱向量送入一个深层且统一的 MLP。让网络自己在高维空间里学习如何搭配高频边缘和低频宏观位置。
    * 结构为：`Linear(35, 128) -> GELU -> Linear(128, 128) -> GELU -> Linear(128, 128)`。
    * 展平二维空间，最终吐出包含极高物理分辨率的初始查询向量 $Q_0$，形状为 `(B, 65536, 128)`。



---

### 🧱 模块 3：级联物理交叉注意力 (Cascaded Cross-Attention)
**使命：完成从 1D 物理空间到 2D 图像空间的非线性雅可比矩阵映射。**

采用**双层级联 (2-Layer Cascaded)** 结构，严格遵守 Transformer 的规范（LayerNorm 先行，残差兜底）：

* **第一层（粗略全局定位）：**
    * $\text{Attn}_1 = \text{MultiHeadCrossAttn}(Q_0, K, V)$
    * $Q'_1 = \text{LayerNorm}(Q_0 + \text{Attn}_1)$
    * $Q_1 = \text{LayerNorm}(Q'_1 + \text{FFN}(Q'_1))$
* **第二层（精准局部校验）：**
    * 拿着更新后的 $Q_1$ 再去查询一次物理电极：
    * $\text{Attn}_2 = \text{MultiHeadCrossAttn}(Q_1, K, V)$
    * $Q'_2 = \text{LayerNorm}(Q_1 + \text{Attn}_2)$
    * $Q_{final} = \text{LayerNorm}(Q'_2 + \text{FFN}(Q'_2))$
* **折叠回图像空间：**
    * 将 `(B, 65536, 128)` 重新 `view` 并 `permute` 成 `(B, 128, 256, 256)` 的二维特征图。
    * *(注：此时特征图上已经能大致看出病灶的物理轮廓，但充满伪影和噪点)*。

---

### 🧱 模块 4：信息瓶颈与双池化 U-Net (Bottleneck & Dual-Pool UNet)
**使命：强制特征提纯，剥离物理伪影，重建极致锐利的异物边界。**

1.  **极速降维 (Information Bottleneck)：**
    * 直接接一个 `1x1 Conv2d`，将通道从 128 压缩到 32。
    * 特征图变为 `(B, 32, 256, 256)`。这一步强迫高维物理特征“挤出水分”，只保留最核心的几何语义，同时大幅减轻 U-Net 的算力压力。
2.  **U-Net 编码器 (Encoder with DualPool)：**
    * 每一层下采样放弃传统的单向 MaxPool，使用我们设计的 `DualPool(x) = Concat[MaxPool(x), -MaxPool(-x)]`。通道数瞬间物理翻倍。
    * **Level 0:** `ConvBlock` 输出 32 通道。
    * **Level 1:** `DualPool` (变64) $\rightarrow$ `ConvBlock` 输出 64 通道。
    * **Level 2:** `DualPool` (变128) $\rightarrow$ `ConvBlock` 输出 128 通道。
    * **Level 3 (Bottleneck):** `DualPool` (变256) $\rightarrow$ `ConvBlock` 输出 256 通道。
3.  **U-Net 解码器 (Decoder)：**
    * 标准的 `ConvTranspose2d` 上采样 $\rightarrow$ 与 Encoder 的同级特征进行 `Concat` $\rightarrow$ `ConvBlock`。
    * 逐级还原回 `(B, 32, 256, 256)`。
4.  **最终输出头 (Output Head)：**
    * 最后的 `1x1 Conv2d` 将 32 通道映射为 3 通道（背景、绝缘体、导电体）。
    * 输出 `(B, 3, 256, 256)` 的 logits。

---

### 💡 训练策略回顾 (配套这套架构的打法)

* **Warm-up 阶段：** 在 Module 3 输出的 128 维特征图上接一个临时的 `1x1 Conv` 直接预测图像算 Loss，逼迫级联注意力层快速学懂物理方程。
* **全参数微调：** 加入 U-Net 联合训练，解开深层形态学的封印。
