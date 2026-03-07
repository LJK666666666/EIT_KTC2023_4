python scripts/generate_data.py --level 3 --num-images 2000 --save-measurements
python scripts/generate_data.py --level 1 --num-images 2 --save-measurements
python scripts/generate_data.py --all-levels --num-images 3 --save-measurements



python scripts/train.py --method fcunet
python scripts/train.py --method postp
python scripts/train.py --method condd --level 3
python scripts/train.py --method postp --max-iters 1
python scripts/train.py --method fcunet --resume results/fcunet_baseline_1/last.pt
python scripts/train.py --method postp --max-iters 1 --batch-size 1 --num-workers 0


