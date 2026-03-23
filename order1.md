python scripts/generate_data.py --level 3 --num-images 2000 --save-measurements
python scripts/generate_data.py --level 1 --num-images 2 --save-measurements
python scripts/generate_data.py --all-levels --num-images 3 --save-measurements

python scripts/train.py --method fcunet
python scripts/train.py --method postp
python scripts/train.py --method condd --level 3
python scripts/train.py --method postp --max-iters 1
python scripts/train.py --method fcunet --resume results/fcunet_baseline_1/last.pt
python scripts/train.py --method postp --max-iters 1 --batch-size 1 --num-workers 0



python scripts/evaluate_all.py --methods fcunet
python scripts/evaluate_all.py --methods postp
python scripts/evaluate_all.py --methods condd



python scripts/generate_data.py --all-levels --num-images 2 --save-measurements
python scripts/generate_val_reco.py
python scripts/train.py --method fcunet --max-iters 1 --batch-size 1 --num-workers 0
python scripts/train.py --method postp --max-iters 1 --batch-size 1 --num-workers 0
python scripts/train.py --method condd --max-iters 1 --batch-size 1 --num-workers 0 --level 1
python scripts/evaluate_all.py --methods fcunet --weights-dir results/fcunet_baseline_1/best.pt
python scripts/evaluate_all.py --methods postp --weights-dir results/postp_baseline_1/best.pt
python scripts/evaluate_all.py --methods condd --weights-dir results/condd_baseline_1/best.pt

python scripts/generate_data.py --level 1 --num-images 1000 --save-measurements --output-dir ../drive/MyDrive/dataset

python scripts/generate_data.py --level 1 --num-images 1000 --save-measurements --output-dir ../drive/MyDrive/dataset --start-idx 1000


python scripts/data_scaling_experiment.py --train-sizes 100 200 400 800 --batch-size 64

python scripts/data_scaling_experiment.py --batch-size 64 --result-dir /content/drive/MyDrive/results

pip install cupy-cuda12x
python scripts/benchmark_data_gen.py --num-samples 5
python scripts/benchmark_data_gen.py --num-samples 40 --workers 4

python scripts/data_scaling_experiment.py --mode postprocess --train-sizes 100 200 400 800 1600 3200 6400

python scripts/generate_data.py --level 1 --num-images 100 --save-measurements --workers 4 --start-idx 1000

python scripts/generate_data.py --level 1 --num-images 9000 --save-measurements --workers 8 --output-dir ../drive/MyDrive/dataset --start-idx 1000

python scripts/npy_to_hdf5.py --input-dir dataset/level_1 --output dataset/level_1/data.h5

python scripts/data_scaling_experiment.py --train-sizes 100 --batch-size 8 --max-iters 1 --hdf5-path dataset/level_1/data.h5

python scripts/generate_data.py --num-images 0 --workers 8 --output-dir ../drive/MyDrive/dataset --start-idx 0 --measurements-only

python scripts/generate_data.py --num-images 400 --workers 4 --start-idx 0 --measurements-only
20s

python scripts/generate_data.py --num-images 800 --workers 2 --start-idx 0 --measurements-only
python scripts/generate_data.py --num-images 800 --workers 2 --start-idx 0 --measurements-only --sys-bias none --output-dir dataset_sim
python scripts/train.py --method dpcaunet --batch-size 8 --hdf5-path dataset/level_1/data.h5
python scripts/train.py --method dpcaunet --batch-size 8 --hdf5-path dataset_sim/level_1/data.h5
python scripts/train.py --method hcdpcaunet --batch-size 8 --hdf5-path dataset/level_1/data.h5
5578mb
python scripts/evaluate_all.py --methods dpcaunet --weights-dir results\dpcaunet_baseline_3

python scripts/generate_data.py --num-images 1600 --workers 2 --measurements-only --sys-bias none --output-dir dataset_sim
168.7s
python scripts/generate_data.py --num-images 1600 --workers 4 --measurements-only --sys-bias none --output-dir dataset_sim
101.2s
python scripts/generate_data.py --num-images 9600 --workers 4 --measurements-only --sys-bias none --output-dir dataset_sim
max35%
python scripts/train.py --method dpcaunet --batch-size 32 --hdf5-path dataset_sim/level_1/data.h5

cd EIT_KTC2023_4 && python scripts/train.py --method dpcaunet --batch-size 32 --hdf5-path dataset_sim/level_1/data.h5 --device tpu
Libtpu version: 0.0.21                                   
Accelerator type: v6e                                    
                                                         
TPU Chips                                       
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID   ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ /dev/vfio/0 │ TPU v6e chip │ 1       │ 19549 │
└─────────────┴──────────────┴─────────┴───────┘
TPU Runtime Utilization                      
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Chip ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 0    │ 13.68 GiB / 31.25 GiB │ 96.88%     │
└──────┴───────────────────────┴────────────┘
TensorCore Utilization              
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Core ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 28.73%                 │
└─────────┴────────────────────────┘
╭─────────── Buffer Transfer Latency Status ────────────╮
│ WARNING: Buffer Transfer Latency metrics unavailable. │
│ Did you start a MULTI_SLICE workload with             │
│ `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?      │
╰───────────────────────────────────────────────────────╯
╭───────────── gRPC TCP Minimum RTT Status ─────────────╮
│ WARNING: gRPC TCP Minimum RTT metrics unavailable.    │
│ Did you start a MULTI_SLICE workload with             │
│ `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?      │
╰───────────────────────────────────────────────────────╯
╭──────────── gRPC TCP Delivery Rate Status ────────────╮
│ WARNING: gRPC TCP Delivery Rate metrics unavailable.  │
│ Did you start a MULTI_SLICE workload with             │
│ `TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434`?      │
╰───────────────────────────────────────────────────────╯

cd EIT_KTC2023_4 && python scripts/train.py --method dpcaunet --batch-size 48 --hdf5-path dataset_sim/level_1/data.h5 --device tpu

