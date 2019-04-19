# coding=utf-8
import os
import inspect
from KukaGymEnv import KukaDiverseObjectEnv
from gym import spaces
from ddpg import DDPG

import time
import numpy as np
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir) # add the '/home/hsk' to the os.sys.path list


class ContinuousDownwardBiasPolicy(object):
  """ Policy which takes continuous actions, and is biased to move down.
  """
  def __init__(self, height_hack_prob=0.9):
    """Initializes the DownwardBiasPolicy.

    Args:
        height_hack_prob: The probability of moving down at every move.
    """
    self._height_hack_prob = height_hack_prob
    self._a_x = spaces.Box(low=-0.05, high=0.05, shape=(1,))
    self._a_y = spaces.Box(low=-0.05, high=0.05, shape=(1,))
    self._a_z = spaces.Box(low=-0.05, high=0.05, shape=(1,))
    self._a_a = spaces.Box(low=np.radians(-90), high=np.radians(90), shape=(1,))

  def choose_action(self):
    """Implements height hack and grasping threshold hack.
    """
    dx = self._a_x.sample()[0]
    dy = self._a_y.sample()[0]
    dz = self._a_z.sample()[0]
    da = self._a_a.sample()[0]

    return np.array([dx, dy, dz, da])


MEMORY_CAPACITY = 50         # memory 容量
MAX_EPISODES = 30000           # learn迭代次数
MAX_EP_STEPS = 50               # 一个回合必须在100个steps内完成任务否则结束
isRENDER = False                # 默认训练期间不渲染


def main():
    ddpg_agent = DDPG(memory_capacity=MEMORY_CAPACITY)      # 初始化一个DDPG智能体
    env = KukaDiverseObjectEnv(renders=isRENDER,
                               isDiscrete=False,
                               maxSteps=MAX_EP_STEPS,
                               removeHeightHack=True,
                               numObjects=3, dv=1.0)
    t1 = time.time()

    ''' The main loop of training '''

    print("\nLearning and Explorating.....")
    learn_graspsuccess = 0.0
    succ_list = np.array([])        # the succession rate list
    steps_list = np.array([])       # step counter
    var = 0.4                       # control exploration 给action加噪声的方差
    for i in range(MAX_EPISODES):
        observe0, done = env.reset(), False  # observe0.shape:(84, 84, 3)
        for j in range(MAX_EP_STEPS):

            full_state0 = env.get_full_state()
            # Add exploration noise
            action = ddpg_agent.choose_action(observe0)
            action[0] = np.clip(np.random.normal(action[0], var), -0.05, 0.05)  # 输出范围(-0.05, 0.05)
            action[1] = np.clip(np.random.normal(action[1], var), -0.05, 0.05)  # 输出范围(-0.05, 0.05)
            action[2] = np.clip(np.random.normal(action[2], var), -0.05, 0.05)  # 输出范围(-0.05, 0.05)
            action[3] = np.clip(np.random.normal(action[3], var), -np.radians(90), np.radians(90))  # 输出范围(-0.05, 0.05)

            observe1, reward, done, info = env.step(action)
            full_state1 = env.get_full_state()
            ddpg_agent.store_transition(full_state0=full_state0,
                                        obs0=observe0,
                                        action=action,
                                        reward=reward,
                                        full_state1=full_state1,
                                        obs1=observe1,
                                        terminal1=done)
            observe0 = observe1
            if info['grasp_success'] == 1:      # 探索阶段 抓取成功 计数器加1
                learn_graspsuccess += 1

            if ddpg_agent.pointer > MEMORY_CAPACITY:
                var *= .9995            # decay the action randomness  减小探索倾向
                ddpg_agent.learn()
            if done:                    # done 指的是尝试完抓取 或者 达到最大steps
                break

        if i % 500 == 0:
            print("%d episodes after,success rate: %f " % (i, learn_graspsuccess / 500))
            succ_list = np.append(succ_list, learn_graspsuccess / 5)
            steps_list = np.append(steps_list,i)
            learn_graspsuccess = 0.

    plt.ylim((0, 100))
    plt.plot(steps_list, succ_list, label='DDPG')
    plt.title('asymmetric_AC+Prior_Replay+n_step_return ')
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('success rate')
    plt.show()

    plt.plot(steps_list, ddpg_agent.n_step_td_loss_list, label='n_step_td_loss')
    plt.plot(steps_list, ddpg_agent.one_step_td_loss_list, label='one_step_td_loss')
    plt.plot(steps_list, ddpg_agent.L2_regular_list, label='L2_regular')

    plt.title('critic loss check')
    plt.legend()
    plt.xlabel('train_Episode')
    plt.ylabel('loss')
    plt.show()
    ddpg_agent.Save()       # save the model
    np.save("n_step_td_loss_list.npy", ddpg_agent.n_step_td_loss_list)
    np.save("one_step_td_loss_list.npy", ddpg_agent.one_step_td_loss_list)
    np.save("L2_regular_list.npy", ddpg_agent.L2_regular_list)
    np.save("succ_list.npy", succ_list)
    np.save("steps_list.npy", steps_list)

    print("learn success number:", learn_graspsuccess)
    print('Final success rate: %f ' % (learn_graspsuccess/MAX_EPISODES))
    print('total Running time: ', (time.time() - t1)/3600.)


if __name__ == '__main__':
    main()
