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

