#!/bin/sh

#SBATCH --job-name=ataxia
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --output=slurm_logs/out_%j.log
#SBATCH --error=slurm_logs/err_%j.log
#SBATCH --time=1-00:00:00

python3 src/trainer.py \
    --log_every 25 \
    --batch_size 64 \
    --epochs 500 \
    --lr 3e-5 \
    --weight_decay 0 \
    --eval_every 50 \
    --save_every 100 \
    --patience 20 \
    --seed 40 \
    --model_type resgcn \
    --layer_num -1 \
    --task regression \
    --log_dir results/gaitgraph \
    --ckpt_path ckpts/resgcn.pt

# additional args - 
# --with_tracking
# --use_mlp
# --model_type stgcn OR resgcn
# --task classification OR --task regression
# --no_shuffle
# --deepnet
# --freeze_encoder
# --overlapping