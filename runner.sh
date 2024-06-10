#!/bin/sh

#SBATCH --job-name=ataxia
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=1-00:00:00

python3 main.py --log_every 10 --batch_size 256 --epochs 1000 --lr 3e-4 \
 --weight_decay 0 --folds 10 --eval_every 10 --save_every 50 \
 --patience 40 --seed 41 --layer_num 3 --with_tracking \
 --ckpt_path "ckpts/st_gcn.kinetics.pt"

# Can also start with nohup
# additional args - 
# --with_tracking
# --use_mlp
# --ensemble