#!/bin/bash
mpirun -np 4 python run.py --batch_size 64 \
 --experiment_name 9_10_12_10 \
 --max_epochs 1000 \
 --max_ep_steps 100 \
 --LAMBDA_predict 0.5 &&
echo "done"