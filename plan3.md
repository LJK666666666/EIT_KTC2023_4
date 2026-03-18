HDF5 数据管线：批量生成 + 转换 + 训练加载
                                                                                                                                            Context
                                                                                                                                        
当前数据以逐样本 .npy 文件存储，多进程生成时不支持 HDF5 输出，且无 HDF5 Dataset 类用于训练。                                             
用户需要：
1. 新脚本将已有 .npy（单样本/多样本）合并写入 .h5
2. generate_data.py 多进程改为内存攒批 → flush .npz → 合并 .h5
3. 训练脚本/评估脚本支持从 .h5 加载数据

修改文件清单

┌────────────────────────────────────┬──────┬────────────────────────────────────────┐
│                文件                │ 操作 │                  说明                  │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ scripts/npy_to_hdf5.py             │ 新建 │ .npy/.npz → .h5 转换工具               │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ scripts/generate_data.py           │ 修改 │ 多进程攒批 + 合并 .h5                  │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/data/sim_dataset.py            │ 修改 │ 新增 FCUNetHDF5Dataset, SimHDF5Dataset │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/data/__init__.py               │ 修改 │ 导出新类                               │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/configs/fcunet_config.py       │ 修改 │ 加 use_hdf5/hdf5_path                  │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/configs/postp_config.py        │ 修改 │ 同上                                   │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/configs/condd_config.py        │ 修改 │ 同上                                   │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/trainers/fcunet_trainer.py     │ 修改 │ build_datasets() 分支 HDF5             │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/trainers/postp_trainer.py      │ 修改 │ 同上                                   │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ src/trainers/condd_trainer.py      │ 修改 │ 同上                                   │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ scripts/train.py                   │ 修改 │ 加 --hdf5-path CLI                     │
├────────────────────────────────────┼──────┼────────────────────────────────────────┤
│ scripts/data_scaling_experiment.py │ 修改 │ 加 --hdf5-path CLI                     │
└────────────────────────────────────┴──────┴────────────────────────────────────────┘

详细方案

1. scripts/npy_to_hdf5.py（新建）

CLI 参数：--input-dir, --output, --data-type {gt,measurements,reco,all}, --dataset-dir(可选，默认当前)

核心逻辑：
- 自动检测 .npy 是单样本 (ndim==单样本维度) 还是批量 (ndim+1)
- 支持 .npz 文件（按 key 拆分写入对应 h5 dataset）
- 打开 h5 用 mode='a'，dataset 用 maxshape=(None,...)，已有则 resize 追加
- chunks=(1, *item_shape) 优化随机访问
- tqdm 进度条

数据类型映射：
gt:           子目录 gt/,           h5 key 'gt',           单样本 ndim=2, dtype float32
measurements: 子目录 measurements/, h5 key 'measurements', 单样本 ndim=1, dtype float64
reco:         子目录 gm_reco/,     h5 key 'reco',          单样本 ndim=3, dtype float32

用法示例：
python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output dataset/level_1/data.h5
python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output /mnt/drive/data.h5 --data-type gt

2. scripts/generate_data.py 多进程改造

新增 CLI: --chunk-size（默认 1000）

新增 worker 函数 _mp_worker_chunked：
- 复用现有样本生成逻辑（提取为 _generate_one_sample() 辅助函数避免重复）
- 内存攒批：gt_buf/meas_buf/reco_buf 列表
- 达到 chunk_size 时 flush 为 _batches/batch_{worker}_{chunk}.npz
- 结束时 flush 剩余

新增 _merge_batches_to_hdf5(base_path)：
- 扫描 _batches/batch_*.npz，逐文件追加到 data.h5
- 完成后删除 _batches/ 目录

main() 多进程分支改造：
if args.workers > 1:
    # 使用 _mp_worker_chunked（支持 GPU 和 CPU）
    with ProcessPoolExecutor(max_workers=w) as pool:
        list(pool.map(_mp_worker_chunked, worker_args))
    # 合并 .npz → .h5
    _merge_batches_to_hdf5(base_path)

注意：--gpu 时仍走单进程路径（GPU 不支持多进程共享）。

3. src/data/sim_dataset.py 新增 HDF5 Dataset 类

FCUNetHDF5Dataset（替代 FCUNetTrainingData）：
- __init__(h5_path, Uref, InvLn, indices=None, augment_noise=True)
- 惰性打开：_h5_file = None，在 __getitem__ 首次调用时 h5py.File(path, 'r')
- 每个 DataLoader worker 进程自动获得独立文件句柄（Windows spawn 安全）
- 返回值与 FCUNetTrainingData 完全一致：(measurements_tensor, gt_onehot_tensor)
- h5py 延迟导入（__init__ 内 import h5py），避免未安装时影响其他类

SimHDF5Dataset（替代 SimData）：
- __init__(h5_path, level, indices=None)
- 同样惰性打开
- 返回值与 SimData 完全一致：(reco_tensor, gt_tensor, level_tensor)

4. 配置文件添加字段

三个 config 文件各加两行：
data.use_hdf5 = False
data.hdf5_path = ''

5. Trainer build_datasets() 分支

三个 trainer 的 build_datasets() 加 HDF5 分支，模式：
use_hdf5 = self.config.data.get('use_hdf5', False)
if use_hdf5:
    from ..data import FCUNetHDF5Dataset  # 或 SimHDF5Dataset
    dataset = FCUNetHDF5Dataset(h5_path, Uelref, solver.InvLn, ...)
else:
    dataset = FCUNetTrainingData(...)  # 原逻辑不变

PostP/CondD 的 hdf5_path 默认从 dataset_base_path 构造：{base_path}/level_{level}/data.h5

6. CLI 脚本适配

scripts/train.py 和 scripts/data_scaling_experiment.py 各加：
parser.add_argument('--hdf5-path', type=str, default=None)
# 在 _apply_overrides / build_experiment_config 中：
if args.hdf5_path is not None:
    config.data.use_hdf5 = True
    config.data.hdf5_path = args.hdf5_path

scripts/evaluate_all.py 无需改动（使用 pipeline，不涉及训练数据集）。

验证

# 1. 转换现有 .npy → .h5
python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output dataset/level_1/data.h5

# 2. 验证 HDF5 内容
python -c "import h5py; f=h5py.File('dataset/level_1/data.h5','r'); print({k:v.shape for k,v in f.items()})"

# 3. 用 HDF5 快速测试训练
python scripts/train.py --method fcunet --hdf5-path dataset/level_1/data.h5 --max-iters 2

# 4. 多进程生成 + 自动合并 HDF5
python scripts/generate_data.py --level 1 --num-images 100 --save-measurements --workers 4 --chunk-size 25
