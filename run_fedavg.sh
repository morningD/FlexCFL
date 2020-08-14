#!/usr/bin/env bash
python3  -u main.py --dataset='shakespeare' --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=20 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='stacked_lstm' \
            --drop_percent=0 \

