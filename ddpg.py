# coding=utf-8

import tensorflow as tf
import numpy as np
from memory import PrioritizedMemory
from baselines.common.mpi_running_mean_std import RunningMeanStd        # update the mean and std dynamically.
import tensorflow.contrib as tc
import baselines.common.tf_util as tf_util
from baselines.common.mpi_adam import MpiAdam

np.random.seed(1)
tf.set_random_seed(1)


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


# --------------- hyper parameters -------------

LR_A = 0.0001       # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.99        # reward discount
TAU = 0.001         # soft replacement
BATCH_SIZE = 32     # memory batch_size in training
N_STEP_RETURN = 5   # N steps return for TD(N) error


# -----------------------------  DDPG ---------------------------------------------

class DDPG(object):
    """ 该类为改进版ddpg算法所有实现.定义一个DDPG智能体（agent）的所有功能

        ---- 类方法 ---------|---------------------------- 描 述 ------------------------------------------|

        '__init__()'         将会实例化一个DDPG智能体.同时生成DDPG的计算图模型,和记忆库,以及一系列超参数.

        'choose_action()'    让智能体根据摄像头采集到的图像,作出行为（action）.

        'learn()'            让智能体完成一次学习： 包括从记忆库中采样一批经验（transitions）->更新'Actor_evaluate'网络
                             -> 更新'Critic_evaluate'网络 -> 更新记忆库优先顺序 -> soft更新 'Actor_target'和
                             ‘Critic_target'网络
        'store_transition()' 在与环境交互的过程中,将经验（transition）存入记忆库

        '_build_a()'         搭建Actor网络结构

        '_build_c()'         搭建Critic网络结构

    """

    def __init__(self, memory_capacity):
        """
        初始化一个DDPG智能体

        :param memory_capacity: 经验库（memory）的容量大小
        """
        self.memory = PrioritizedMemory(capacity=memory_capacity, alpha=0.8)
        self.pointer = 0                        # memory 计数器　
        self.sess = tf.InteractiveSession()     # 创建一个默认会话
        self.lambda_1step = 1.0                 # 1_step_return_loss的权重
        self.lambda_nstep = 1.0                 # n_step_return_loss的权重

        # 定义 placeholders
        self.observe_Input = tf.placeholder(tf.float32, [None, 84, 84, 3], name='observe_Input')
        self.observe_Input_ = tf.placeholder(tf.float32, [None, 84, 84, 3], name='observe_Input_')
        self.f_s = tf.placeholder(tf.float32, [None, 24], name='full_state_input')
        self.f_s_ = tf.placeholder(tf.float32, [None, 24], name='fill_state_input_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.n_step_steps = tf.placeholder(tf.float32, shape=(None, 1), name='nstep_reached')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        # 预处理:  包括图像（obs）的预处理 和 full_state的预处理.
        #         预处理方法为'z-score 标准化'. 经过处理的数据符合标准正态分布.
        #         目的为提升收敛速度和效果.

        with tf.variable_scope('obs_rms'):
            self.obs_rms = RunningMeanStd(shape=(84, 84, 3))
        with tf.variable_scope('state_rms'):
            self.state_rms = RunningMeanStd(shape=(24,))
        with tf.name_scope('obs_preprocess'):
            self.normalized_observe_Input = tf.clip_by_value(
                normalize(self.observe_Input, self.obs_rms), -10., 10.)
            self.normalized_observe_Input_ = tf.clip_by_value(
                normalize(self.observe_Input_, self.obs_rms), -10., 10.)
        with tf.name_scope('state_preprocess'):
            self.normalized_state = normalize(self.f_s, self.state_rms)
            self.normalized_state_ = normalize(self.f_s_, self.state_rms)

        # The calculate operations for Actor
        with tf.variable_scope('Actor'):
            # action = actor_evaluate(obs)
            self.action = self.build_actor(self.normalized_observe_Input, scope='eval', trainable=True)
            # action_ = actor_target(obs_)
            action_ = self.build_actor(self.normalized_observe_Input_, scope='target', trainable=False)

        # The calculate operations for Critic
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor.

            # q = critic_evaluate(full_state, action)
            q = self.build_critic(self.normalized_state, self.action, scope='eval', trainable=True)

            # q(s_,a_) = critic_target(full_state_, action_)
            #          = critic_target(full_state_, actor_target(obs_))
            #          ---- paper 公式5
            q_ = self.build_critic(self.normalized_state_, action_, scope='target', trainable=False)

        # Collect networks parameters. It would make it more easily to manage them.
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # The soft_update of target_net.
        self.soft_replace_a = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.at_params, self.ae_params)]
        self.soft_replace_c = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.ct_params, self.ce_params)]

        """ 
            计算 '1_step_q_target' 和 'n_step_target_q'.
            在学习过程中, 先喂记忆库采样得到的 '1_step_batch' 和 'n_step_batch', 将两个'target'分别求出.
            再执行critic优化器
        """
        self.q_target = self.R + GAMMA * q_
        self.n_step_target_q = self.R + (1. - self.terminals1) * tf.pow(GAMMA, self.n_step_steps) * q_
        self.td_error = tf.square(self.q_target - q)
        self.nstep_td_error = tf.square(self.n_step_target_q - q)

        td_losses = tf.reduce_mean(tf.multiply(self.ISWeights, self.td_error)) * self.lambda_1step
        n_step_td_losses = tf.reduce_mean(tf.multiply(self.ISWeights, self.nstep_td_error)) * self.lambda_nstep
        critic_losses = td_losses + n_step_td_losses + tf.contrib.layers.apply_regularization(
                                                                    tf.contrib.layers.l2_regularizer(0.001),
                                                                    weights_list=self.ce_params)
        # L_critic = lambda_1_step * L_1_step + lambda_n_step * L_n_step + lambda2

        # maximize the q. Baslines也是求q均值.再对actor网络参数求梯度,并向梯度方向更新.(paper 公式6)
        a_loss = - tf.reduce_mean(q)

        # Setting optimizer for Actor and Critic
        self.critic_grads = tf_util.flatgrad(critic_losses, self.ce_params)
        self.critic_optimizers = MpiAdam(var_list=self.ce_params,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

        #  init_target net-work with evaluate net-params
        self.init_a_t = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
        self.init_c_t = [tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)]
        self.sess.run(self.init_a_t)
        self.sess.run(self.init_c_t)

        # 保存模型
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, obs):
        """
        让智能体根据摄像头采集到的图像(obs),作出行为（action）.
        :param obs: 输入图像. shape = (84, 84, 3). dtype = uint8.
        :return: action: dx,dy,dz,da.    dx -> (-0.05,0.05)
                                         dy -> (-0.05,0.05)
                                         dz -> (-0.05,0.05)
                                         da -> (-pi/2,pi/2)
        """
        obs = obs.astype(dtype=np.float32)
        return self.sess.run(self.action, {self.observe_Input: obs[np.newaxis, :]})[0]

    def Save(self):
        self.saver.save(self.sess, save_path="/home/hsk/Documents/kuka_grasp_ddpg/model/model.ckpt")

    def load(self):
        ckpt = tf.train.get_checkpoint_state("/home/hsk/Documents/kuka_grasp_ddpg/model/")
        self.saver.restore(self.sess, save_path=ckpt.model_checkpoint_path)

    def learn(self):
        """
                先从记忆库中采样得到 '1_step_batch' 和 'n_step_batch'.
            -> 将两个'target'分别求出.后面记忆库更新优先顺序时还需要.
            -> 执行critic优化器, 执行actor优化器.
            -> 更新记忆库优先顺序权重
            -> soft_更新'target_网络'
        """
        batch, n_step_batch, percentage = self.memory.sample_rollout(
                            batch_size=BATCH_SIZE,
                            nsteps=N_STEP_RETURN,
                            beta=0.4,
                            gamma=GAMMA
                            )
        # calculate the target_q for 1 step and N steps separately
        target_q_1step = self.sess.run(
            self.q_target,
            feed_dict={
                self.observe_Input_: batch['obs1'],
                self.f_s_: batch['full_states1'],
                self.R: batch['rewards'],
            })
        n_step_target_q = self.sess.run(
            self.n_step_target_q,
            feed_dict={ self.terminals1: n_step_batch["terminals1"],
                        self.n_step_steps: n_step_batch["step_reached"],
                        self.R: n_step_batch['rewards'],
                        self.observe_Input_: n_step_batch['obs1'],
                        self.f_s_: n_step_batch['full_states1']})
        # calculate td_errors and grads of critic
        td_error, nstep_td_error, critic_grads = self.sess.run(
                [self.td_error, self.nstep_td_error, self.critic_grads],
                            feed_dict={
                                self.q_target: target_q_1step,
                                self.n_step_target_q: n_step_target_q,
                                self.f_s: batch['full_states0'],
                                self.action: batch['actions'],
                                self.ISWeights: batch['weights']
                                })
        # update critic
        self.critic_optimizers.update(critic_grads, stepsize=LR_C)

        # update actor
        self.sess.run(self.atrain, {self.observe_Input: batch['obs0'],
                                    self.f_s: batch['full_states0']})

        self.memory.update_priorities(batch['idxes'], td_errors=(td_error + nstep_td_error))

        # soft target replacement
        self.sess.run(self.soft_replace_a)
        self.sess.run(self.soft_replace_c)

    def store_transition(self,
                         obs0,
                         action,
                         reward,
                         obs1,
                         full_state0,
                         full_state1,
                         terminal1,
                         demo=False):
        """
        在与环境交互的过程中,将经验（transition）存入记忆库.

        :param obs0:   观察到图像. shape = (84, 84, 3). dtype = uint8.
        :param action: 行为（action）: dx,dy,dz,da
        :param reward: 观察到图像（obs0）条件下, 执行行为（action）后,立即得到的奖励.
        :param obs1:   观察到图像（obs0）条件下, 执行行为（action）后,观察到图像（obs1）.
        :param full_state0: 观察到图像（obs0）时, 模拟器内的完全状态
        :param full_state1: 观察到图像（obs0）条件下, 执行行为（action）后,模拟器内的完全状态.
        :param terminal1:   是否游戏结束
        :param demo:        是否使用教师经验
        """
        obs0 = obs0.astype(np.float32)
        obs1 = obs1.astype(np.float32)
        full_state0 = full_state0.astype(np.float32)
        full_state1 = full_state1.astype(np.float32)

        if demo:
            self.memory.append_demonstration(full_state0, obs0, action, reward,
                                             full_state1, obs1, terminal1)
        else:
            self.memory.prior_append(full_state0=full_state0,
                                     obs0=obs0,
                                     action=action,
                                     reward=reward,
                                     full_state1=full_state1,
                                     obs1=obs1,
                                     terminal1=terminal1)

        self.obs_rms.update(np.array([obs0]))
        self.state_rms.update(np.array([full_state0]))
        self.pointer += 1

    def build_actor(self, observe_input, scope, trainable):
        """

        :param observe_input: 观察图像.shape = (None, 84, 84, 3)
        :param scope:         命名空间
        :param trainable:     是否能被训练. bool值.
        :return: action: dx,dy,dz,da.    dx -> (-0.05,0.05)
                                         dy -> (-0.05,0.05)
                                         dz -> (-0.05,0.05)
                                         da -> (-pi/2,pi/2)
        """
        with tf.variable_scope(scope):
            # conv *3 + flatten
            normalizer_fn = tc.layers.layer_norm
            conv_1 = tc.layers.conv2d(inputs=observe_input, num_outputs=32, kernel_size=3,
                                      stride=2, padding='valid', trainable=trainable, normalizer_fn=normalizer_fn)

            conv_2 = tc.layers.conv2d(inputs=conv_1, num_outputs=32, kernel_size=3,
                                      stride=2, padding='valid', trainable=trainable,normalizer_fn=normalizer_fn)

            conv_3 = tc.layers.conv2d(inputs=conv_2, num_outputs=32, kernel_size=3,
                                      stride=2, padding='valid', trainable=trainable,normalizer_fn=normalizer_fn)

            conv_4 = tc.layers.conv2d(inputs=conv_3, num_outputs=32, kernel_size=9,
                                      stride=2, padding='valid', trainable=trainable,normalizer_fn=normalizer_fn)

            flatten_layer = tf.layers.flatten(conv_4)

            # dense *2  units = 200
            dense_1 = tf.layers.dense(flatten_layer, units=256, activation=None, trainable=trainable)
            dense_1 = tc.layers.layer_norm(dense_1, center=True, scale=True)
            dense_1 = tf.nn.relu(dense_1)

            dense_2 = tf.layers.dense(dense_1, units=256, activation=None, trainable=trainable)
            dense_2 = tc.layers.layer_norm(dense_2, center=True, scale=True)
            dense_2 = tf.nn.relu(dense_2)

            dense_3 = tf.layers.dense(dense_2, units=256, activation=None, trainable=trainable)
            dense_3 = tc.layers.layer_norm(dense_3, center=True, scale=True)
            dense_3 = tf.nn.relu(dense_3)

            dense_3 = tf.layers.dense(dense_3, units=256, activation=None, trainable=trainable)
            dense_3 = tc.layers.layer_norm(dense_3, center=True, scale=True)
            dense_3 = tf.nn.relu(dense_3)

            action_output = tf.layers.dense(dense_3, units=4, activation=tf.nn.tanh,
                                            kernel_initializer=tf.initializers.random_uniform(minval=-0.003,
                                                                                              maxval=0.003), trainable=trainable)
            #输出(1,4)
            action_output = action_output * np.array([0.05, 0.05, 0.05, np.radians(90)])
            # dx a[0] (-0.05,0.05)
            # dy a[1] (-0.05,0.05)
            # dz a[2] (-0.05,0.05)
            # da a[3] (-pi/2,pi/2)

            return action_output

    def build_critic(self, f_s, a, scope, trainable):
        """
        搭建'Asymmetric_critic'网络

        :param f_s:       完全信息  (None,24)
        :param a:         输入行为 （None,4）
        :param scope:     命名空间
        :param trainable: 是否能被训练. bool值
        :return: q(s,a)
        """
        with tf.variable_scope(scope):
            # 和a连起来
            if a.shape == (4,):  # a有(4,)  和 (BATCH_SIZE,4) 两种情况
                a = a[np.newaxis, :]  # (4,)->(1,4)
            connected_f_a = tf.concat([f_s, a], axis=1)

            dense_1 = tf.layers.dense(connected_f_a, units=256, activation=None, trainable=trainable)
            dense_1 = tc.layers.layer_norm(dense_1, center=True, scale=True)
            dense_1 = tf.nn.relu(dense_1)

            dense_2 = tf.layers.dense(dense_1, units=256, activation=None, trainable=trainable)
            dense_2 = tc.layers.layer_norm(dense_2, center=True, scale=True)
            dense_2 = tf.nn.relu(dense_2)

            dense_3 = tf.layers.dense(dense_2, units=256, activation=None, trainable=trainable)
            dense_3 = tc.layers.layer_norm(dense_3, center=True, scale=True)
            dense_3 = tf.nn.relu(dense_3)

            dense_4 = tf.layers.dense(dense_3, units=256, activation=None, trainable=trainable)
            dense_4 = tc.layers.layer_norm(dense_4, center=True, scale=True)
            dense_4 = tf.nn.relu(dense_4)

            q = tf.layers.dense(dense_4, units=1, kernel_initializer=tf.initializers.random_uniform(minval=-0.0003,
                                                                                              maxval=0.0003),trainable=trainable)
            # Q(s,a) 输出一个[None,1]
            return q
