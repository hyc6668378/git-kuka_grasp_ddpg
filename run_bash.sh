#!/bin/bash
mpirun -np 5 python run.py --batch_size 64 \
 --max_ep_steps 100 \
 --evaluation \
 --experiment_name no_predict \
 --memory_size 10000 \
 --max_epochs 123 \
 --use_DDPGfD \
 --priority \
 --LAMBDA_predict 0.0 &&

echo "done"