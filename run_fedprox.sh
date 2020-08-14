#!/usr/bin/env bash
python3  -u main.py --dataset='nist' --optimizer='fedprox'  \
            --learning_rate=0.003 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=0 \
            --mu=1 \
