#!/bin/bash
mpirun -np 3 python run.py --memory_size 10000 \
 --batch_size 64 \
 --experiment_name test_mpi_ddpg \
 --max_ep_steps 20 \
 --total_timesteps 10000 \
 --nb_rollout_steps 1 &&
echo "done"