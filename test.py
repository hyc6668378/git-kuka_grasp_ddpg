import argparse
import os
from mpi4py import MPI
import tensorflow as tf
from env.KukaGymEnv import KukaDiverseObjectEnv
from algorithm.ddpg import DDPG


def common_arg_parser():
    parser = argparse.ArgumentParser()

    # common argument
    parser.add_argument('--memory_size',    type=int, default=2019, help="MEMORY_CAPACITY. default = 2019")
    parser.add_argument('--inter_learn_steps', type=int, default=5, help="一个step中agent.learn()的次数. default = 3")
    parser.add_argument('--experiment_name',   type=str, default='no_name', help="实验名字")
    parser.add_argument('--batch_size',    type=int, default=16, help="batch_size. default = 16")
    parser.add_argument('--max_ep_steps',    type=int, default=50, help="一个episode最大长度. default = 50")
    parser.add_argument('--seed',    type=int, default=0, help="random seed. default = 0")
    parser.add_argument('--isRENDER',  action="store_true", help="渲染GUI .")
    parser.add_argument('--total_timesteps', type=int, default=int(1e+6), help="The timestep of whole training. default = 1e+6")
    parser.add_argument("--noise_target_action", help="noise target_action for Target policy smoothing", action="store_true")
    parser.add_argument("--nb_rollout_steps", type=int, default=5, help="The timestep of whole training. default = 5")


    # priority memory replay
    parser.add_argument("--priority", action="store_true", help="priority memory replay")
    parser.add_argument('--alpha', type=float, default=0.2, help="priority degree")
    parser.add_argument("--turn_beta",  action="store_true", help="turn the beta from 0.6 to 1.0")

    # n_step_return
    parser.add_argument("--use_n_step", help="use n_step_loss", action="store_true")
    parser.add_argument('--n_step_return', type=int, default=5, help="n step return. default = 5")

    # DDPGfD
    parser.add_argument("--use_DDPGfD",  action="store_true", help="train with Demonstration")
    parser.add_argument('--Demo_CAPACITY', type=int, default=3000, help="The number of demo transitions. default = 2000")
    parser.add_argument('--PreTrain_STEPS', type=int, default=2000, help="The steps for PreTrain. default = 2000")
    parser.add_argument("--LAMBDA_BC", type=int, default=100, help="behavior clone weight. default = 100.0")

    # TD3
    parser.add_argument("--use_TD3", help="使用TD3 避免过估计", action="store_true")
    parser.add_argument("--policy_delay", type=int, default=2, help="policy update delay w.r.t critic update. default = 2")

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args
class A(object):
    def __init__(self):
        self.a = 1
def check(a):
    a.a += 1

def train(seed, experiment_name, **kwargs):
    # print('max_episodes:', max_episodes)
    # print('use_TD3:', use_TD3)
    rank = MPI.COMM_WORLD.Get_rank()
    seed = seed + 2019 * rank
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=30,
                               removeHeightHack=True,
                               numObjects=3, dv=1.0)
    kwargs['obs_space'] = env.observation_space
    kwargs['action_space'] = env.action_space

    agent = DDPG(rank,  **kwargs, experiment_name=experiment_name)
    os.system("clear")
    for _ in range(4):
        print(agent.pi(env.observation_space.sample()))
    aa = A()
    print(aa.a)
    check(aa)
    print("--",aa.a)
    print("ok!")


if __name__ == '__main__':
    args = common_arg_parser()
    train(**args)

