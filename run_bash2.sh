#!/bin/bash
python run.py -p \
 --memory_size 10000 \
 --batch_size 64 \
 --alpha 0.2 \
 --use_n_step \
 --experiment_name demo_no_TD3 \
 --PreTrain_STEPS 3000 \
 --Demo_CAPACITY 2000 \
 --max_episodes 15000 \
 --noise_target_action \
 --LAMBDA_BC 5000 &&
echo "done"