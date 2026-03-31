# KTC2023 评分加速说明

## 背景

仓库原始评分主要使用 [`src/evaluation/scoring.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py) 中的：

- `scoring_function`
- `FastScoringFunction`

其中：

- `scoring_function` 是官方风格实现，内部使用 `scipy.signal.convolve2d` 做大核二维高斯卷积，数值准确，但速度极慢。
- `FastScoringFunction` 使用可分离一维卷积替代二维卷积，数学指标定义不变，速度明显提升。

在实际测试中，官方评分已经成为评估脚本的主要瓶颈。

## 本次加速内容

本次新增了 Torch CUDA 快速评分实现：

- [`src/evaluation/scoring_torch.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring_torch.py)

核心内容：

- `TorchFastScorer`
  - 使用 `torch.nn.functional.conv2d`
  - 将二维高斯卷积拆成两个一维卷积
  - 支持 GPU 加速
  - 支持批量评分

- `fast_score_auto(groundtruth, reconstruction, device=None)`
  - 自动选择评分后端
  - 若 `cuda` 可用，则优先使用 Torch CUDA
  - 否则退回 [`FastScoringFunction`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/evaluation/scoring.py)

- `fast_score_batch_auto(groundtruths, reconstructions, device=None)`
  - 批量评分版本
  - `cuda` 可用时使用 `TorchFastScorer.score_batch`
  - 否则逐张调用 `FastScoringFunction`

## 数学关系

这次加速没有改变 score 的定义。

评分仍然是：

1. 将标签图拆成两张二值 mask
   - 类 `1`
   - 类 `2`
2. 分别计算 SSIM
3. 最后取平均

区别只在于卷积实现：

- 官方版：完整二维卷积
- 快速版：可分离一维卷积
- Torch 版：在 Torch 中实现可分离一维卷积，并支持 CUDA

因此：

- 指标定义保持不变
- 与官方实现存在极小浮点误差
- 但在实验中误差量级很小，可忽略

## Benchmark 脚本

新增独立 benchmark 脚本：

- [`scripts/benchmark_scoring.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/benchmark_scoring.py)

作用：

- 单样本比较官方评分、SciPy 快速评分、Torch CUDA 快速评分
- 输出时间、分数和相对官方的误差

示例命令：

```bash
python scripts/benchmark_scoring.py --official-runs 1 --fast-runs 3 --torch-runs 20
```

结果目录示例：

- [`results/scoring_benchmark_1`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/scoring_benchmark_1)
- [`results/scoring_benchmark_2`](/D:/010_CodePrograms/E/EIT_KTC2023_4/results/scoring_benchmark_2)

## 实测结果

在当前机器上，单样本 benchmark 的一组结果如下：

- `official`: `92015 ms`
- `fast_scipy`: `127 ms`
- `fast_torch_cuda`: `3.15 ms`
- `fast_torch_batched_cuda`: `2.46 ms`
- `fast_torch_cached_cuda`: `2.32 ms`

对应结论：

- 相比官方评分，SciPy 快速版约 `723x` 加速
- 相比官方评分，Torch CUDA 版约 `4e4x` 加速
- Torch CUDA 版比分离卷积 SciPy 快速版仍快约 `50x`

数值差异示例：

- `official_score = 0.910397`
- `fast_torch_cached_cuda = 0.910396`
- 误差约 `5.4e-7`

## 已接入 Torch CUDA 快速评分的入口

### 1. `evaluate_all.py`

[`scripts/evaluate_all.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/evaluate_all.py) 目前支持：

- `--score-mode official`
- `--score-mode fast`
- `--score-mode torch`
- `--score-device auto/cpu/cuda`

示例：

```bash
python scripts/evaluate_all.py --methods fcunet --weights-dir results --score-mode torch --score-device cuda
```

### 2. FCUNet trainer

[`src/trainers/fcunet_trainer.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/trainers/fcunet_trainer.py)

以下评估已改成自动优先使用 Torch CUDA 快速评分：

- `_validate_challenge`
- `evaluate_test`

内部调用：

- `fast_score_batch_auto(...)`

### 3. Diffusion trainer

[`src/diffusion/trainer.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/diffusion/trainer.py)

挑战集评分已改为：

- `fast_score_auto(...)`

### 4. SAE 重建可视化

[`scripts/visualize_sae_reconstruction.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/visualize_sae_reconstruction.py)

在传入：

```bash
--compute-score
```

时，已改为自动优先使用 Torch CUDA 快速评分。

## 当前统一策略

除 [`scripts/evaluate_all.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/scripts/evaluate_all.py) 外，其他主要评分入口现在统一采用：

- CUDA 可用：Torch CUDA 快速评分
- CUDA 不可用：退回 `FastScoringFunction`

这样可以：

- 避免每个脚本单独维护评分模式参数
- 保持默认行为尽可能快

## 仍保留官方评分的原因

官方评分仍然保留，主要用于：

- 严格复现实验
- 与官方结果逐项比对
- 在最终报分前做一致性确认

建议日常使用：

- 大规模评估：Torch CUDA 快速评分
- 最终严格核对：官方评分

## 后续可选清理

仓库里目前还保留旧重复实现：

- [`src/ktc_methods/scoring_fast.py`](/D:/010_CodePrograms/E/EIT_KTC2023_4/src/ktc_methods/scoring_fast.py)

主流程已经不再依赖它。后续可以：

1. 删除该文件
2. 或明确标记为 deprecated

以避免仓库中出现多份 fast scorer 并行维护。
