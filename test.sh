#!/bin/bash
python test.py \
 --experiment_name seg_with_predict \
 --use_segmentation_Mask \
 --seed 2 \
 --max_ep_steps 100 &&
echo "done"