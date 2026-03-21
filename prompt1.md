仿真合并
验证集来源

我现在有1000条level_1的仿真数据，我想在level_1上训练fcunet，训练数据为仿真数据的100/200/400/800，验证集、测试集为仿真数据的100，查看测试集效果随数据规模的变化曲线，你觉得可行吗？


@scripts\data_scaling_experiment.py 中的学习率调度机制和早停机制是否会同步调整？

为什么 @scripts\data_scaling_experiment.py 结果保存的文件夹名全部为 @results\fcunet_scaling_n100_{num} ，n100不应该根据--train-sizes动态调整吗？

你可以创建一个脚本对 @results\fcunet_scaling_n{num}_{num} 中之前的训练记录'training_log.json'进行patience=15的早停截断吗？

我发现训练过程中验证时间似乎比训练时间还要久，这是为什么？score计算是不是比较费时？


创建新脚本实现将 @dataset 中已生成的数据改为.h5格式（支持命令行参数指定数据路径，因为要在colab上使用）。
将 @scripts\generate_data.py 中生成数据改成.h5格式，支持向.h5文件中继续添加新数据。
修改 @scripts\data_scaling_experiment.py @scripts\train.py @scripts\evaluate_all.py 支持对.h5格式的数据进行加载处理。

创建新脚本实现将 @dataset 中已生成的数据（支持单样本.npy和多样本.npy）继续写入.h5格式数据文件（可能不存在，也可能已有文件内容）中（支持命令行参数指定数据路径，因为要在colab上使用）。
将 @scripts\generate_data.py 中生成数据机制改成：1、比如 8 个 Worker（多进程）在内存里疯狂算数据。2、算出来的结果不要立刻写盘，而是暂存在每个 Worker 自己内存的 List 里。3、当某个 Worker 的 List 攒够了 1000 个样本（大约也就占用一两百 MB 内存），它就一次性把这 1000 个样本打包成一个稍微大一点的 .npy 或 .npz 写到硬盘里。4、等所有进程跑完，你得到了 100 个包含了 1000 个样本的中型文件。此时调用前面写的新脚本，花几分钟时间，极其顺滑地把这 100 个文件一口气导入到最终的 .h5 文件中。
修改 @scripts\data_scaling_experiment.py @scripts\train.py @scripts\evaluate_all.py 支持对.h5格式的数据进行加载处理。

=写入云盘的通信瓶颈


请尝试根据 @GUIDE\data_generation_optimization.md 中**K. GPU 预处理共轭梯度法（CG）替代 CPU 直接求解**的思路，对 @scripts\generate_data.py 中的forward步骤进行Batched PCG加速。先不要对原脚本进行修改，创建一个新脚本进行尝试，将优化后效果和当前效果进行对比，并不断进行优化尝试，直到效果符合预期为止。


转存机制

gm_reco中的文件内容是什么？为什么比gt和measurements大这么多？

@scripts\npy_to_hdf5.py 是如何将.npy文件写入.h5文件的？它是直接将.npy文件接在.h5已有最大索引的最后面吗？还是会按照文件名写入对应索引？

KTC2023有没有提供32个电极的位置信息？或者其他测量条件信息？
神经网络的预测结果是真实电导率还是3值标签？


=数据质量（数量、形状）
=数据数量
=sim to real(噪声)
神经网络

把固定系统误差的模拟设置为命令行参数选项，以便课程学习或其他模拟需求

将每个电极的76个测量值和角度余弦、正弦合起来作为78个输入值，乘上一个78*(d-2)的K/V得到d-2维向量后再拼接角度余弦、正弦，然后将每个位置的归一化横纵坐标、与中心点距离、角度余弦正弦共5个值输入一个MLP中得到d-5维向量后再拼接归一化横纵坐标、与中心点的归一化距离、角度余弦正弦得到q。(拆分成多头)将q点乘k除以根号d再softmax后乘上V得到最终d张N*N特征图。然后卷积、最大最小池化、卷积、最大最小池化等等，然后反卷积等等，并且上下采样中同尺寸特征图进行连接，即U-Net，最后得到一张预测的图像。是这样吧？

添加代码，实现 @update7.md 中的模型架构，并接入到 @scripts\train.py @scripts\evaluate_all.py 中。

第1个电极在水箱正上方，左右边界为 \pm 2.8125°，然后沿逆时针方向每隔11.25°依次排列后续电极，电极与电导率像素的角度表示最好能对齐。
数据集中从level_1到level_7会逐级移除相应数量的电极对，需要兼容输入数据缺失的情况，当前神经网络是否能鲁棒处理这种情况？

