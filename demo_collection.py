# coding=utf-8

from env.KukaGymEnv import KukaDiverseObjectEnv
import argparse
import numpy as np
from multiprocessing import Pool
import os

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--max_ep_steps',    type=int, default=100, help="一个episode最大长度. default = 100")
    return  parser

parser = common_arg_parser()
args = parser.parse_args()

def collect_worker(worker_index):
    print (str(worker_index)+" start!")
    i = worker_index * 300
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=args.max_ep_steps,
                               blockRandom=0.4,
                               removeHeightHack=True,
                               use_low_dim_obs=False,
                               use_segmentation_Mask=True,
                               numObjects=1, dv=1.0)
    while 1:
        obs0, done = env.reset(), False
        f_s0 = env._low_dim_full_state()
        for j in range(args.max_ep_steps):
            action = env.demo_policy()

            obs1, reward, done, info = env.step(action)
            if info['is_success']:
                print('success in %d transition'%(i))
            f_s1 = env._low_dim_full_state()
            demo_transitions = {'obs0': obs0,
                                'f_s0': f_s0,
                                'action': action,
                                'obs1': obs1,
                                'reward': reward,
                                'f_s1': f_s1,
                                'terminal1': done}
            obs0 = obs1
            f_s0 = f_s1

            demo_tran_file_path = 'demo_with_segm/demo%d.npy'%(i)
            np.save(demo_tran_file_path, demo_transitions, allow_pickle=True)

            i = i + 1
            if done:
                break

            if i >= (worker_index+1) * 300  :
                return

if __name__ == '__main__':
    # 开12个进程 一起收集demo
    print('Parent process %s.' % os.getpid())
    p = Pool(12)

    for k in range(12):
        p.apply_async(collect_worker, args=(k,))
    p.close()
    p.join()
    print('All subprocesses done.')