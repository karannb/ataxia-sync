#!/bin/sh

#SBATCH --job-name=ataxia
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=1-00:00:00

python3 main.py --log_every 10 --batch_size 128 --epochs 1000 --lr 7e-5 \
 --weight_decay 0 --folds 10 --eval_every 10 --save_every 50 --with_tracking \
 --patience 40 --seed 41 --layer_num 9 --task classification \
 --ckpt_path "ckpts/st_gcn.kinetics.pt"

# Can also start with nohup
# additional args - 
# --with_tracking
# --use_mlp
# --task classification OR --task regression