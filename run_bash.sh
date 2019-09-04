#!/bin/bash
mpirun -np 4 python run.py --batch_size 64 \
 --experiment_name grasp_dense_reward \
 --max_epochs 1000000 \
 --priority \
 --max_ep_steps 20 \
 --nb_rollout_steps 1 &&
echo "done"