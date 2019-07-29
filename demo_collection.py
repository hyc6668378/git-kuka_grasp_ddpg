# coding=utf-8

from KukaGymEnv import KukaDiverseObjectEnv
import numpy as np
import argparse

def common_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--alpha', type=float, default=0.2, help="priority degree")
    parser.add_argument('--memory_size',    type=int, default=2019, help="MEMORY_CAPACITY. default = 2019")
    parser.add_argument('--inter_learn_steps', type=int, default=5, help="一个step中agent.learn()的次数. default = 3")
    parser.add_argument('--experiment_name',   type=str, default='no_name', help="实验名字")
    parser.add_argument('--batch_size',    type=int, default=16, help="batch_size. default = 16")
    parser.add_argument('--max_ep_steps',    type=int, default=50, help="一个episode最大长度. default = 50")
    parser.add_argument('--seed',    type=int, default=0, help="random seed. default = 0")
    parser.add_argument('--isRENDER',  action="store_true", help="渲染GUI .")
    parser.add_argument("--turn_beta",  help="turn the beta from 0.6 to 1.0", action="store_true")
    parser.add_argument("--use_n_step", help="use n_step_loss", action="store_true")
    parser.add_argument('--n_step_return',    type=int, default=5, help="n step return. default = 5")

    return  parser

parser = common_arg_parser()
args = parser.parse_args()

env = KukaDiverseObjectEnv(renders=args.isRENDER,
                           isDiscrete=False,
                           maxSteps=args.max_ep_steps,
                           removeHeightHack=True,
                           numObjects=3, dv=1.0)
def main():
    print("\nDemo_Collection.....")
    i = 0
    while 1:
        obs0, done = env.reset(), False
        f_s0 = env.get_full_state()
        for j in range(args.max_ep_steps):
            action = env.demo_policy()

            obs1, reward, done, info = env.step(action)
            f_s1 = env.get_full_state()
            demo_transitions = {'obs0': obs0,
                                'f_s0': f_s0,
                                'action': action,
                                'obs1': obs1,
                                'reward': reward,
                                'f_s1': f_s1,
                                'terminal1': done}
            obs0 = obs1
            f_s0 = f_s1

            demo_tran_file_path = 'all_demo/demo%d.npy'%(i)
            np.save(demo_tran_file_path, demo_transitions, allow_pickle=True)

            i = i + 1
            if done:
                break
            if i % 500 == 0:
                print("%d demos has been collected " % (i))
        if i > 2000:
            break

if __name__ == '__main__':
    main()