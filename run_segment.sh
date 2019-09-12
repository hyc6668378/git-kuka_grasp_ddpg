#!/bin/bash
mpirun -np 5 python run.py --batch_size 64 \
 --use_segmentation_Mask \
 --max_ep_steps 100 \
 --evaluation \
 --experiment_name seg_with_predict \
 --memory_size 10000 \
 --max_epochs 123 \
 --use_DDPGfD \
 --priority \
 --LAMBDA_predict 1.0 &&

echo "done" &&

shutdown