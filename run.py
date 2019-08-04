# coding=utf-8
import os
from KukaGymEnv import KukaDiverseObjectEnv
from ddpg import DDPG
from OUNoise import OUNoise
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import tensorflow as tf
import random

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
    parser.add_argument("--turn_beta",  action="store_true", help="turn the beta from 0.6 to 1.0")
    parser.add_argument("--use_n_step", help="use n_step_loss", action="store_true")
    parser.add_argument('--n_step_return', type=int, default=5, help="n step return. default = 5")
    parser.add_argument('--Demo_CAPACITY', type=int, default=3000, help="The number of demo transitions. default = 2000")
    parser.add_argument('--PreTrain_STEPS', type=int, default=2000, help="The steps for PreTrain. default = 2000")
    parser.add_argument('--max_episodes', type=int, default=8000, help="The Max episodes. default = 8000")
    parser.add_argument("--noise_target_action", help="noise target_action for Target policy smoothing", action="store_true")
    parser.add_argument("--LAMBDA_BC", type=int, default=100, help="behavior clone weitht. default = 100.0")
    parser.add_argument("--policy_delay", type=int, default=2, help="policy update delay w.r.t critic update. default = 2")
    parser.add_argument("--use_TD3", help="使用TD3 避免过估计", action="store_true")

    return parser

parser = common_arg_parser()
args = parser.parse_args()


agent = DDPG(memory_capacity=args.memory_size, batch_size=args.batch_size,
             prioritiy=args.priority, noise_target_action=args.noise_target_action,
             alpha=args.alpha, use_n_step=args.use_n_step, n_step_return=args.n_step_return,
             is_training=True, LAMBDA_BC=args.LAMBDA_BC, policy_delay=args.policy_delay,
             use_TD3=args.use_TD3, experiment_name=args.experiment_name)

env = KukaDiverseObjectEnv(renders=args.isRENDER,
                           isDiscrete=False,
                           maxSteps=args.max_ep_steps,
                           removeHeightHack=True,
                           numObjects=3, dv=1.0)

Noise = OUNoise(size=4, mu=0, theta=0.05, sigma=0.25)

def mkdir(experiment_name):
    folder = os.path.exists("logs/"+experiment_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs("logs/"+experiment_name)

    folder = os.path.exists("result/" + experiment_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs("result/" + experiment_name)

    folder = os.path.exists("model/" + experiment_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs("model/" + experiment_name)

def set_global_seeds(myseed):

    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def save_all(succ_list, steps_list, demo_per):
    agent.Save()  # save the model
    np.save("result/" + args.experiment_name + "/" + args.experiment_name + "_succ_list.npy", succ_list)
    np.save("result/" + args.experiment_name + "/" + args.experiment_name + "_steps_list.npy", steps_list)
    np.save("result/" + args.experiment_name + "/" + args.experiment_name + "_demo_percentage", demo_per)

def Noise_Action(action):
    noise = Noise.sample()
    action = np.clip( action + noise,
                      a_min=[-0.05, -0.05, -0.05, -np.radians(90)],
                      a_max=[0.05, 0.05, 0.05, np.radians(90)] )
    return action

def plot(succ_list, steps_list):

    plt.figure(figsize=(21, 11))
    plt.ylim((0, 100))
    plt.plot(steps_list, succ_list, label='success_rate')
    plt.title(args.experiment_name)
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('success rate')
    plt.savefig('result/'+args.experiment_name+'/'+args.experiment_name+'.png')

def demo_collect(Demo_CAPACITY):

    print("\n Demo Collecting...")

    for i in tqdm(range(Demo_CAPACITY)):
        demo_tran_file_path = 'all_demo/demo%d.npy' % (i)
        transition = np.load(demo_tran_file_path, allow_pickle=True)
        transition = transition.tolist()
        agent.store_transition(full_state0=transition['f_s0'],
                               obs0=transition['obs0'],
                               action=transition['action'],
                               reward=transition['reward'],
                               full_state1=transition['f_s1'],
                               obs1=transition['obs1'],
                               terminal1=transition['terminal1'],
                               demo = True)
    print(" Demo Collection completed.")

def preTrain(PreTrain_STEPS):
    print("\n PreTraining ...")

    for _ in tqdm(range(PreTrain_STEPS)):
        agent.learn()

    print(" PreTraining completed.")

def train(max_episodes):

    learn_grasp_success = 0.0
    for episoed in tqdm(range(max_episodes)):
        obs0, done = env.reset(), False
        f_s0 = env.get_full_state()
        for j in range(args.max_ep_steps):

            action = agent.pi( f_s0 ) # low dim obs
            action = Noise_Action( action )

            obs1, reward, done, info = env.step( action )
            f_s1 = env.get_full_state()

            agent.store_transition(full_state0=f_s0,
                                   obs0=obs0,
                                   action=action,
                                   reward=reward,
                                   full_state1=f_s1,
                                   obs1=obs1,
                                   terminal1=done)
            obs0, f_s0 = obs1, f_s1

            if info['grasp_success'] == 1:
                learn_grasp_success += 1

            if agent.pointer > args.memory_size:
                for _ in range(args.inter_learn_steps):
                    agent.learn()
            if done:
                break

        # 将一个episode的结果打印到 tensorboard
        agent.save_episoed_result(done, episoed)

        # Noise decay
        Noise.theta = np.linspace(0.05, 0.0, max_episodes)[i]
        Noise.sigma = np.linspace(0.25, 0.0, max_episodes)[i]
        if args.turn_beta:
            agent.beta = np.linspace(0.6, 1.0, max_episodes)[i]


def main():
    t1 = time.time()

    # 生成实验文件夹
    mkdir(args.experiment_name)

    set_global_seeds(args.seed)

    os.system("clear")

    demo_collect( args.Demo_CAPACITY )

    preTrain( args.PreTrain_STEPS )

    train( args.max_episodes )

    save_all(agent.demo_percent)

    print("total Running time:{:.2f}(h) ".format((time.time() - t1)/3600.))

if __name__ == '__main__':
    main()
