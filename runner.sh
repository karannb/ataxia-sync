#!/bin/sh

python3 main.py --log_every 10 --batch_size 256 --epochs 1000 --lr 3e-5 \
 --weight_decay 0.0 --folds 10 --eval_every 10 --save_every 50 \
 --patience 10 --seed 42 --layer_num 4 \
 --ckpt_path "ckpts/st_gcn.kinetics.pt" > logs/out.txt 2> logs/err.txt &

# Can also start with nohup
# additional args - 
# --with_tracking
# --use_mlp
# --ensemble