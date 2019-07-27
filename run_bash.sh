#!/bin/bash
python run.py -p --memory_size 2000 --inter_learn_steps 3 --batch_size 64\
 --alpha 0.4 --use_n_step --experiment_name 1k_3step_64batch_5_step_return_alpha0.2_beta0.4 &&
echo "done"