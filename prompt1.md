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

