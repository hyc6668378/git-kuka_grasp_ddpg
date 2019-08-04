#!/bin/bash
python run.py -p \
 --memory_size 3000 \
 --isRENDER \
 --batch_size 64 \
 --alpha 0.2 \
 --use_n_step \
 --experiment_name low_dim_obs \
 --PreTrain_STEPS 3000 \
 --Demo_CAPACITY 1000 \
 --max_episodes 15000 \
 --use_TD3 \
 --LAMBDA_BC 5 &&
echo "done"