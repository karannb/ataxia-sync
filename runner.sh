#!/bin/sh

python3 main.py --log_every 10 --batch_size 256 --epochs 1000 --lr 7e-5 \
 --weight_decay 0.0 --folds 10 --eval_every 10 --save_every 50 \
 --patience 40 --seed 42 --layer_num 1 \
 --ckpt_path "ckpts/st_gcn.kinetics.pt" > logs/out2.txt 2> logs/err2.txt &

# Can also start with nohup
# additional args - 
# --with_tracking
# --use_mlp
# --ensemble