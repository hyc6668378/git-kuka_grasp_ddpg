#!/bin/bash
python run.py -p --memory_size 1000 --inter_learn_steps 3 \
 --alpha 0.2 --use_n_step --experiment_name 1k_3step_5_step_return_alpha0.2_beta0.6_ &&
echo "done"