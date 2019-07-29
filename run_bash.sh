#!/bin/bash
python run.py -p --memory_size 10000 --batch_size 64\
 --alpha 0.2 --use_n_step --experiment_name demo_64batch \
 --PreTrain_STEPS 3000 --Demo_CAPACITY 2000 --max_episodes 15000 &&
echo "done"