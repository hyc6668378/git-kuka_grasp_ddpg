#!/bin/bash
python run.py -p --memory_size 8000 --batch_size 64\
 --alpha 0.2 --use_n_step --isRENDER --experiment_name demo_64batch \
 --PreTrain_STEPS 3000 --max_episodes 10000 &&
echo "done"