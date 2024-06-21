#!/bin/sh

#SBATCH --job-name=ataxia
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=1-00:00:00

python3 main.py --log_every 25 --batch_size 128 --epochs 1000 --lr 7e-5 \
 --weight_decay 0 --folds 10 --eval_every 50 --save_every 200 \
 --patience 40 --seed 45 --layer_num -2 --task classification \
 --ckpt_path "ckpts/st_gcn.kinetics.pt" --no_shuffle --with_tracking

# Can also start with nohup
# additional args - 
# --with_tracking
# --use_mlp
# --task classification OR --task regression
# --no_shuffle