# coding=utf-8
import os
from KukaGymEnv import KukaDiverseObjectEnv
from ddpg import DDPG
from OUNoise import OUNoise
import time
import numpy as np
import matplotlib.pyplot as plt

MEMORY_CAPACITY = 20197         # memory 容量
max_ep_steps = 50               # 一个回合必须在100个steps内完成任务否则结束
isRENDER = False                # 默认训练期间不渲染
Inter_Learn_Steps = 5

ddpg_agent = DDPG(memory_capacity=MEMORY_CAPACITY)  # 初始化一个DDPG智能体
env = KukaDiverseObjectEnv(renders=isRENDER,
                           isDiscrete=False,
                           maxSteps=max_ep_steps,
                           removeHeightHack=True,
                           numObjects=3, dv=1.0)

Noise = OUNoise(size=4, mu=0, theta=0.05, sigma=0.25)

def Noise_Action(action):
    noise = Noise.sample()
    action = np.clip( action + noise,
                      a_min=[-0.05, -0.05, -0.05, -np.radians(90)],
                      a_max=[0.05, 0.05, 0.05, np.radians(90)] )
    return action

def plot(succ_list, steps_list):

    plt.figure(figsize=(21, 11))
    plt.ylim((0, 100))
    plt.plot(steps_list, succ_list, label='DDPG')
    plt.title('asymmetric_AC ')
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('success rate')
    plt.show()


    plt.title('critic loss check')
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('loss')
    plt.savefig('result/asymmetric_AC.png')

def train(max_episodes):
    succ_list = np.array([])  # the succession rate list
    steps_list = np.array([])  # step counter
    learn_graspsuccess = 0.0
    for i in range(max_episodes):
        obs0, done = env.reset(), False
        f_s0 = env.get_full_state()
        for j in range(max_ep_steps):

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

            if ddpg_agent.pointer > MEMORY_CAPACITY:
                for _ in range(Inter_Learn_Steps):
                    ddpg_agent.learn()
            if done:  # done 指的是尝试完抓取 或者 达到最大steps
                break
        # Noise decay
        Noise.theta = np.linspace(0.05, 0.0, max_episodes)[i]
        Noise.sigma = np.linspace(0.25, 0.0, max_episodes)[i]

        if i % 500 == 0:
            print("%d episodes after,success rate: %f " % (i, learn_graspsuccess / 500))
            succ_list = np.append(succ_list, learn_graspsuccess / 5)
            steps_list = np.append(steps_list,i)
            learn_graspsuccess = 0.
    return succ_list, steps_list

def main():
    t1 = time.time()
    os.system("clear")
    succ_list, steps_list = train(max_episodes=30000)

    plot(succ_list, steps_list)

    ddpg_agent.Save()       # save the model

    np.save("result/succ_list.npy", succ_list)
    np.save("result/steps_list.npy", steps_list)

    print('total Running time: ', (time.time() - t1)/3600.)

if __name__ == '__main__':
    main()
