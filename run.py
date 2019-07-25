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
parser = argparse.ArgumentParser()
parser.add_argument('--memory_size',    type=int, default=2019, help="MEMORY_CAPACITY. default = 2019")
parser.add_argument('--inter_learn_steps', type=int, default=5, help="一个step中agent.learn()的次数. default = 3")
parser.add_argument('--experiment_name',   type=str, default='no_name', help="实验名字")
parser.add_argument('--batch_size',    type=int, default=128, help="batch_size. default = 128")
parser.add_argument('--max_ep_steps',    type=int, default=50, help="一个episode最大长度. default = 50")
parser.add_argument('--seed',    type=int, default=0, help="random seed. default = 0")
parser.add_argument('--isRENDER',    type=bool, default=False, help="是否渲染. default = False")
args = parser.parse_args()


ddpg_agent = DDPG(memory_capacity=args.memory_size, batch_size=args.batch_size)  # 初始化一个DDPG智能体
env = KukaDiverseObjectEnv(renders=args.isRENDER,
                           isDiscrete=False,
                           maxSteps=args.max_ep_steps,
                           removeHeightHack=True,
                           numObjects=3, dv=1.0)

Noise = OUNoise(size=4, mu=0, theta=0.05, sigma=0.25)

def set_global_seeds(myseed):

    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def save_all(succ_list, steps_list):
    ddpg_agent.Save()  # save the model
    np.save("result/" + args.experiment_name + "_succ_list.npy", succ_list)
    np.save("result/" + args.experiment_name + "_steps_list.npy", steps_list)

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
    plt.savefig('result/'+args.experiment_name+'.png')

def train(max_episodes):
    succ_list = np.array([])  # the succession rate list
    steps_list = np.array([])  # step counter
    learn_graspsuccess = 0.0
    for i in tqdm(range(max_episodes)):
        obs0, done = env.reset(), False
        f_s0 = env.get_full_state()
        for j in range(args.max_ep_steps):

            action = ddpg_agent.choose_action(obs0)
            action = Noise_Action(action)

            obs1, reward, done, info = env.step(action)
            f_s1 = env.get_full_state()

            ddpg_agent.store_transition(full_state0=f_s0,
                                        obs0=obs0,
                                        action=action,
                                        reward=reward,
                                        full_state1=f_s1,
                                        obs1=obs1,
                                        terminal1=done)
            obs0 = obs1
            f_s0 = f_s1

            if info['grasp_success'] == 1:  # 探索阶段 抓取成功 计数器加1
                learn_graspsuccess += 1

            if ddpg_agent.pointer > args.memory_size:
                for _ in range(args.inter_learn_steps):
                    ddpg_agent.learn()
            if done:  # done 指的是尝试完抓取 或者 达到最大steps
                break
        # Noise decay
        Noise.theta = np.linspace(0.05, 0.0, max_episodes)[i]
        Noise.sigma = np.linspace(0.25, 0.0, max_episodes)[i]

        if i % 500 == 0:
            succ_list = np.append(succ_list, learn_graspsuccess / 5)
            steps_list = np.append(steps_list,i)
            learn_graspsuccess = 0.
            save_all(succ_list, steps_list)
    return succ_list, steps_list

def main():
    t1 = time.time()
    os.system("clear")
    set_global_seeds(args.seed)
    succ_list, steps_list = train(max_episodes=10000)

    save_all(succ_list, steps_list)

    plot(succ_list, steps_list)
    print('total Running time: ', (time.time() - t1)/3600.)

if __name__ == '__main__':
    main()
