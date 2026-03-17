仿真合并
验证集来源

我现在有1000条level_1的仿真数据，我想在level_1上训练fcunet，训练数据为仿真数据的100/200/400/800，验证集、测试集为仿真数据的100，查看测试集效果随数据规模的变化曲线，你觉得可行吗？


@scripts\data_scaling_experiment.py 中的学习率调度机制和早停机制是否会同步调整？

为什么 @scripts\data_scaling_experiment.py 结果保存的文件夹名全部为 @results\fcunet_scaling_n100_{num} ，n100不应该根据--train-sizes动态调整吗？

你可以创建一个脚本对 @results\fcunet_scaling_n{num}_{num} 中之前的训练记录'training_log.json'进行patience=15的早停截断吗？

我发现训练过程中验证时间似乎比训练时间还要久，这是为什么？score计算是不是比较费时？
