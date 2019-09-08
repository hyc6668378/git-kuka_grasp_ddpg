#!/bin/bash
mpirun -np 1 python run.py --batch_size 64 \
 --experiment_name 9_8_20_39 \
 --max_epochs 10000 \
 --priority \
 --max_ep_steps 100 \
 --inter_learn_steps 10 \
 --nb_rollout_steps 1 &&
echo "done"