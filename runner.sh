#!/bin/sh

#SBATCH --job-name=ataxia
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=1-00:00:00

python3 main.py --log_every 25 --batch_size 64 --epochs 500 --lr 3e-5 \
 --weight_decay 0 --folds 10 --eval_every 50 --save_every 100 \
 --patience 20 --seed 40 --layer_num 6 --task regression --log_dir results \
 --ckpt_path ckpts/st_gcn.kinetics.pt --non_overlapping

# additional args - 
# --with_tracking
# --use_mlp
# --task classification OR --task regression
# --no_shuffle
# --deepnet
# --freeze_encoder
# --non_overlapping