# coding=utf-8

import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd        # update the mean and std dynamically.
import tensorflow.contrib as tc
from functools import partial
from algorithm.My_toolkit import normalize
from stable_baselines.common import tf_util
from stable_baselines.common.mpi_adam import MpiAdam

conv2_ = partial(tc.layers.conv2d, padding='valid', activation_fn=None)


# --------------- hyper parameters -------------

LR_A = 1e-4       # learning rate for actor
LR_C = 1e-3        # learning rate for critic
GAMMA = 0.99        # reward discount
TAU = 0.001         # soft replacement

# -----------------------------  DDPG ---------------------------------------------

class DDPG(object):
    def __init__(self, rank, batch_size, priority, use_n_step=False, n_step_return=5,
                 LAMBDA_BC = 100, LAMBDA_predict=0.5, policy_delay=1, use_TD3=False, experiment_name='none', Q_value_range=(-250, 5), **kwargs):

        self.batch_size = batch_size
        self.use_prioritiy = priority
        self.n_step_return = n_step_return
        self.use_n_step = use_n_step
        self.LAMBDA_BC = LAMBDA_BC
        self.LAMBDA_predict = LAMBDA_predict
        self.use_TD3 = use_TD3
        self.experiment_name = experiment_name
        self.Q_value_range = Q_value_range # 限制q的范围,防止过估计.
        self.demo_percent = [] # demo 在 sample中所占比例

        self.pointer = 0                        # memory 计数器　
        self.num_timesteps = 0                  # steps for tensorboard
        self.lambda_1_step = 0.5                 # 1_step_return_loss的权重
        self.lambda_n_step = 0.5                 # n_step_return_loss的权重
        self.beta = 0.6

        # actor 比 critic 更新频率小
        self.policy_delay_iterate = 0
        self.policy_delay = policy_delay

        self._setup_model(rank, **kwargs)

    def _setup_model(self, rank, memory_size, alpha,
                     obs_space, action_space, full_state_space, noise_target_action, **kwargs):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf_util.single_threaded_session(graph=self.graph)
            if self.use_prioritiy:
                from algorithm.priority_memory import PrioritizedMemory
                self.memory = PrioritizedMemory(capacity=memory_size, alpha=alpha)
            else:
                from algorithm.memory import Memory
                self.memory = Memory(limit=memory_size, action_shape=action_space.shape,
                                     observation_shape=obs_space.shape,
                                     full_state_shape=full_state_space.shape)
            # 定义 placeholders
            self.observe_Input = tf.placeholder(tf.float32, [None]+list(obs_space.shape), name='observe_Input')
            self.observe_Input_ = tf.placeholder(tf.float32, [None]+list(obs_space.shape), name='observe_Input_')
            self.f_s = tf.placeholder(tf.float32, [None]+list(full_state_space.shape), name='full_state_Input')
            self.f_s_ = tf.placeholder(tf.float32, [None]+list(full_state_space.shape), name='fill_state_Input_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')
            self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
            self.n_step_steps = tf.placeholder(tf.float32, shape=(None, 1), name='n_step_reached')
            self.q_demo = tf.placeholder(tf.float32, [None, 1], name='Q_of_actions_from_memory')
            self.come_from_demo = tf.placeholder(tf.float32, [None, 1], name='Demo_index')
            self.action_memory = tf.placeholder(tf.float32, [None]+list(action_space.shape), name='actions_from_memory')

            with tf.variable_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=obs_space.shape)

            with tf.variable_scope('state_rms'):
                self.state_rms = RunningMeanStd(shape=full_state_space.shape)

            with tf.name_scope('obs_preprocess'):
                self.normalized_observe_Input = tf.clip_by_value(
                    normalize(self.observe_Input, self.obs_rms), -5., 5.)
                self.normalized_observe_Input_ = tf.clip_by_value(
                    normalize(self.observe_Input_, self.obs_rms), -5., 5.)

            with tf.name_scope('state_preprocess'):
                self.normalized_f_s0 = normalize(self.f_s, self.state_rms)
                self.normalized_f_s1 = normalize(self.f_s_, self.state_rms)

            with tf.variable_scope('Actor'):
                self.action, f_s_predict = self.build_actor(self.normalized_observe_Input,
                                               scope='eval', trainable=True, full_state_dim=full_state_space.shape[0])
                self.action_, _ = self.build_actor(self.normalized_observe_Input_,
                                                scope='target', trainable=False,full_state_dim=full_state_space.shape[0])

                # Target policy smoothing, by adding clipped noise to target actions
                if noise_target_action:
                    epsilon = tf.random_normal(tf.shape(self.action_), stddev=0.007)
                    epsilon = tf.clip_by_value(epsilon, -0.01, 0.01)
                    a2 = self.action_ + epsilon
                    noised_action_ = tf.clip_by_value(a2, -1, 1)
                else:
                    noised_action_ = self.action_

            with tf.variable_scope('Critic'):
                # Q值都要被clip 防止过估计.
                self.q_1 = tf.clip_by_value(
                    self.build_critic(self.normalized_f_s0, self.action,
                                      scope='eval_1', trainable=True),
                    self.Q_value_range[0], self.Q_value_range[1])

                q_1_ = self.build_critic(self.normalized_f_s1, noised_action_,
                                         scope='target_1', trainable=False)

                if self.use_TD3:
                    q_2 = tf.clip_by_value(
                        self.build_critic(self.normalized_f_s0, self.action,
                                          scope='eval_2', trainable=True),
                        self.Q_value_range[0], self.Q_value_range[1])

                    q_2_ = self.build_critic(self.normalized_f_s1, noised_action_,
                                             scope='target_2', trainable=False)

            # Collect networks parameters. It would make it more easily to manage them.
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
            self.ce1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_1')
            self.ct1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_1')

            if self.use_TD3:
                self.ce2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_2')
                self.ct2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_2')

            with tf.variable_scope('Soft_Update'):
                self.soft_replace_a = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                                       zip(self.at_params, self.ae_params)]
                self.soft_replace_c = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                                       zip(self.ct1_params, self.ce1_params)]
                if self.use_TD3:
                    self.soft_replace_c += [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in
                                            zip(self.ct2_params, self.ce2_params)]

            # critic 的误差 为 (one-step-td 误差 + n-step-td 误差 + critic_online 的L2惩罚)
            # TD3: critic一共有4个, 算两套 critic的误差, 秀儿.
            with tf.variable_scope('Critic_Lose'):
                if self.use_TD3:
                    min_q_ = tf.minimum(q_1_, q_2_)
                else:
                    min_q_ = q_1_

                self.q_target = self.R + (1. - self.terminals1) * GAMMA * min_q_
                if self.use_n_step:
                    self.n_step_target_q = self.R + (1. - self.terminals1) * tf.pow(GAMMA, self.n_step_steps) * min_q_
                    cliped_n_step_target_q = tf.clip_by_value(self.n_step_target_q, self.Q_value_range[0],
                                                              self.Q_value_range[1])

                cliped_q_target = tf.clip_by_value(self.q_target, self.Q_value_range[0], self.Q_value_range[1])

                self.td_error_1 = tf.abs(cliped_q_target - self.q_1)
                if self.use_TD3:
                    self.td_error_2 = tf.abs(cliped_q_target - q_2)

                if self.use_n_step:
                    self.nstep_td_error_1 = tf.abs(cliped_n_step_target_q - self.q_1)
                    if self.use_TD3:
                        self.nstep_td_error_2 = tf.abs(cliped_n_step_target_q - q_2)

                L2_regular_1 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.001),
                                                                      weights_list=self.ce1_params)
                if self.use_TD3:
                    L2_regular_2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.001),
                                                                          weights_list=self.ce2_params)

                one_step_losse_1 = tf.reduce_mean(
                    tf.multiply(self.ISWeights, tf.square(self.td_error_1))) * self.lambda_1_step
                if self.use_TD3:
                    one_step_losse_2 = tf.reduce_mean(
                        tf.multiply(self.ISWeights, tf.square(self.td_error_2))) * self.lambda_1_step

                if self.use_n_step:
                    n_step_td_losses_1 = tf.reduce_mean(
                        tf.multiply(self.ISWeights, tf.square(self.nstep_td_error_1))) * self.lambda_n_step
                    c_loss_1 = one_step_losse_1 + n_step_td_losses_1 + L2_regular_1

                    if self.use_TD3:
                        n_step_td_losses_2 = tf.reduce_mean(
                            tf.multiply(self.ISWeights, tf.square(self.nstep_td_error_2))) * self.lambda_n_step
                        c_loss_2 = one_step_losse_2 + n_step_td_losses_2 + L2_regular_2
                else:
                    c_loss_1 = one_step_losse_1 + L2_regular_1

                    if self.use_TD3:
                        c_loss_2 = one_step_losse_2 + L2_regular_2

            # actor 的 loss 为 最大化q(s,a) 最小化行为克隆误差.
            # (只有demo的transition 且 demo的action 比 actor生成的action q_1(s,a)高的时候 才会有克隆误差)
            with tf.variable_scope('Actor_lose'):
                Is_worse_than_demo = self.q_1 < self.q_demo
                Is_worse_than_demo = tf.cast(Is_worse_than_demo, tf.float32)
                worse_than_demo = tf.cast(tf.reduce_sum(Is_worse_than_demo), tf.int8)

                # 算action误差 我用的是平方和, 也有人用均方误差 reduce_mean. 其实都可以.
                # 我的action本来都是很小的数.
                action_diffs = Is_worse_than_demo * tf.reduce_sum(
                    self.come_from_demo * tf.square(self.action - self.action_memory), 1, keepdims=True)

                L_BC = self.LAMBDA_BC * tf.reduce_sum(action_diffs)
                auxiliary_predict_loss = self.LAMBDA_predict * tf.reduce_mean( tf.square(f_s_predict - self.f_s))
                a_loss = - tf.reduce_mean(self.q_1) + L_BC + auxiliary_predict_loss

            # Setting optimizer for Actor and Critic
            with tf.variable_scope('Critic_Optimizer'):
                if self.use_TD3:
                    self.critic_grads_1 = tf_util.flatgrad(loss=c_loss_1, var_list=self.ce1_params)
                    self.critic_grads_2 = tf_util.flatgrad(loss=c_loss_2, var_list=self.ce2_params)

                    self.critic_optimizer_1 = MpiAdam(var_list=self.ce1_params, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-08)
                    self.critic_optimizer_2 = MpiAdam(var_list=self.ce2_params, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-08)
                else:
                    self.critic_grads = tf_util.flatgrad(loss = c_loss_1, var_list=self.ce1_params)
                    self.critic_optimizer = MpiAdam(var_list=self.ce1_params, beta1=0.9, beta2=0.999,
                            epsilon=1e-08)
            with tf.variable_scope('Actor_Optimizer'):
                self.actor_grads = tf_util.flatgrad(a_loss, self.ae_params)
                self.actor_optimizer = MpiAdam(var_list=self.ae_params, beta1=0.9, beta2=0.999,
                                               epsilon=1e-08)
            with self.sess.as_default():
                self._initialize(self.sess)

            # 保存模型
            var_list = tf.global_variables()
            print("var_list!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            for v in var_list:
                print(v)
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
            self.writer = tf.summary.FileWriter("logs/" + self.experiment_name + "/DDPG_"+ str(rank), self.graph)
            # TensorBoard summary
            self.a_summary = tf.summary.merge([tf.summary.scalar('a_loss', a_loss, family='actor'),
                                               tf.summary.scalar('L_BC', L_BC, family='actor'),
                                               tf.summary.scalar('worse_than_demo', worse_than_demo, family='actor'),
                                               tf.summary.scalar('auxiliary_predict_loss', auxiliary_predict_loss, family='actor')])

            if self.use_TD3:
                self.c_summary = tf.summary.merge([tf.summary.scalar('c_loss_1', c_loss_1, family='critic'),
                                                   tf.summary.scalar('c_loss_2', c_loss_2, family='critic')])
            else:
                self.c_summary = tf.summary.merge([tf.summary.scalar('c_loss_1', c_loss_1, family='critic')])

            # episode summary
            self.episode_cumulate_reward = tf.placeholder(tf.float32, name='episode_cumulate_reward')
            self.episoed_length = tf.placeholder(tf.int16, name='episode_cumulate_reward')
            self.success_or_not = tf.placeholder(tf.int8, name='episode_cumulate_reward')

            self.eval_episode_cumulate_reward = tf.placeholder(tf.float32, name='episode_cumulate_reward')
            self.eval_episoed_length = tf.placeholder(tf.int16, name='episode_cumulate_reward')
            self.eval_success_or_not = tf.placeholder(tf.int8, name='episode_cumulate_reward')

            self.episode_summary = tf.summary.merge([
                tf.summary.scalar('episode_cumulate_reward', self.episode_cumulate_reward, family='episoed_result'),
                tf.summary.scalar('episoed_length', self.episoed_length, family='episoed_result'),
                tf.summary.scalar('success_or_not', self.success_or_not, family='episoed_result'),])

            self.eval_episode_summary = tf.summary.merge([
                tf.summary.scalar('eval_episode_cumulate_reward', self.eval_episode_cumulate_reward, family='Eval_episoed_result'),
                tf.summary.scalar('eval_episoed_length', self.eval_episoed_length, family='Eval_episoed_result'),
                tf.summary.scalar('eval_success_or_not', self.eval_success_or_not, family='Eval_episoed_result'), ])

    def _initialize(self, sess):
        """
        initialize the model parameters and optimizers

        :param sess: (TensorFlow Session) the current TensorFlow session
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.actor_optimizer.sync()
        self.critic_optimizer.sync()

        #  init_target net-work with evaluate net-params
        init_a_t = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
        init_c_t = [tf.assign(t, e) for t, e in zip(self.ct1_params, self.ce1_params)]
        if self.use_TD3:
            init_c_t += [tf.assign(t, e) for t, e in zip(self.ct2_params, self.ce2_params)]
        self.sess.run([init_a_t, init_c_t])

    def pi(self, obs):
        obs = obs.astype(dtype=np.float32)
        return self.sess.run(self.action, {self.observe_Input: obs[np.newaxis, :]})[0]

    def Save(self):
        # 只存权重,不存计算图.
        self.saver.save(self.sess, save_path="model/" + self.experiment_name + "/model.ckpt", write_meta_graph=False)

    def load(self):
        self.saver.restore(self.sess, save_path="model/" + self.experiment_name + "/model.ckpt")

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    def save_episoed_result(self, epi_cumulate_reward, episoed_length, success_or_not, episodes):
        s = self.sess.run(self.episode_summary,
                          feed_dict={self.episode_cumulate_reward: epi_cumulate_reward,
                                     self.episoed_length: episoed_length,
                                     self.success_or_not: success_or_not})

        self.writer.add_summary(s, global_step=int(episodes))

    def save_eval_episoed_result(self, eval_epi_reward, eval_episoed_length, eval_success_or_not, eval_episodes):
        eval_s = self.sess.run(self.eval_episode_summary,
                          feed_dict={self.eval_episode_cumulate_reward: eval_epi_reward,
                                     self.eval_episoed_length: eval_episoed_length,
                                     self.eval_success_or_not: eval_success_or_not})

        self.writer.add_summary(eval_s, global_step=int(eval_episodes))

    def learn(self, learn_step):
        if self.use_prioritiy:
            batch, n_step_batch, percentage = self.memory.sample_rollout(
                batch_size=self.batch_size,
                nsteps=self.n_step_return,
                beta=self.beta,
                gamma=GAMMA
            )
            self.demo_percent.append(float(percentage))
        else:
            batch = self.memory.sample( self.batch_size )

        one_step_target_q = self.sess.run(
            self.q_target,
            feed_dict={
                self.observe_Input_: batch['obs1'], # low dim input
                self.R: batch['rewards'],
                self.terminals1: batch['terminals1'],
                self.f_s_: batch['f_s1']
                       })

        if self.use_TD3:
            opt = [self.td_error_1, self.td_error_2, self.critic_grads_1, self.critic_grads_2, self.c_summary, self.q_1]
        else:
            opt = [self.td_error_1, self.critic_grads, self.c_summary, self.q_1]

        if self.use_prioritiy and self.use_n_step:
            n_step_target_q = self.sess.run(
                self.n_step_target_q,
                feed_dict={self.terminals1: n_step_batch["terminals1"],
                           self.n_step_steps: n_step_batch["step_reached"],
                           self.R:  n_step_batch['rewards'],
                           self.observe_Input_: n_step_batch['obs1'],
                           self.f_s_: n_step_batch['f_s1']
                           })
            res = self.sess.run(
                opt, feed_dict={
                    self.q_target: one_step_target_q,
                    self.n_step_target_q: n_step_target_q,
                    self.f_s: batch['f_s0'],
                    self.action: batch['actions'],
                    self.ISWeights: batch['weights']
                })
        else:
            res = self.sess.run(
                opt, feed_dict={
                    self.q_target: one_step_target_q,
                    self.f_s: batch['f_s0'],
                    self.action: batch['actions'],
                    self.ISWeights: batch['weights']
                })

        # critic update
        if self.use_TD3:
            td_error_1, td_error_2, critic_grads_1, critic_grads_2, c_s, q_demo = res
            td_error = (td_error_1 + td_error_2) / 2.0

            self.critic_optimizer_1.update(critic_grads_1, learning_rate=LR_C)
            self.critic_optimizer_1.update(critic_grads_1, learning_rate=LR_C)
        else:
            td_error, critic_grads, c_s, q_demo = res
            self.critic_optimizer.update(critic_grads, LR_C)
        self.sess.run(self.soft_replace_c)

        # actor update
        if self.policy_delay_iterate % self.policy_delay == 0:
            actor_grads, a_s, = self.sess.run([self.actor_grads, self.a_summary],
                                   {self.observe_Input: batch['obs0'],
                                    self.q_demo: q_demo,
                                    self.f_s: batch['f_s0'],
                                    self.come_from_demo: batch['demos'],
                                    self.action_memory: batch['actions']})

            self.actor_optimizer.update(actor_grads, LR_A)
            self.sess.run(self.soft_replace_a)
            self.writer.add_summary(a_s, learn_step)

        # update priority
        if self.use_prioritiy:
            self.memory.update_priorities(batch['idxes'], td_error )

        self.writer.add_summary(c_s, learn_step)
        self.policy_delay_iterate += 1

    def store_transition(self,
                         obs0,
                         action,
                         reward,
                         obs1,
                         full_state0,
                         full_state1,
                         terminal1,
                         demo = False):
        obs0 = obs0.astype(np.float32)
        obs1 = obs1.astype(np.float32)
        full_state0 = full_state0.astype(np.float32)
        full_state1 = full_state1.astype(np.float32)
        if demo:
            self.memory.append_demo( obs0=obs0, f_s0=full_state0, action=action, reward=reward,
                                     obs1=obs1, f_s1=full_state1, terminal1=terminal1)
        else:
            self.memory.append( obs0=obs0, f_s0=full_state0, action=action, reward=reward,
                            obs1=obs1, f_s1=full_state1, terminal1=terminal1)

        # 增量式的更新observe的 mean, std
        self.obs_rms.update(np.array([obs0]))
        self.obs_rms.update(np.array([obs1]))
        self.state_rms.update(np.array([full_state0]))
        self.state_rms.update(np.array([full_state1]))

        self.pointer += 1

    def build_actor(self, observe_input, scope, trainable, full_state_dim):
        fc_a = partial(tf.layers.dense, activation=None, trainable=trainable)
        conv2_a = partial( conv2_, trainable=trainable)
        relu = partial(tf.nn.relu)
        with tf.variable_scope(scope):
            # conv -> relu
            net = relu(conv2_a( observe_input, 32, 7, 4 ))
            net = relu(conv2_a( net, 64, 5, 2 ))
            net = relu(conv2_a( net, 64, 3, 2 ))
            net = relu(conv2_a( net, 64, 3, 1 ))

            net = tf.layers.flatten(net)
            net = relu(fc_a( net, 128 + full_state_dim ))
            net, f_s_predict = tf.split( net, [128, full_state_dim], 1)

            net = relu(fc_a( net, 128 ))
            net = relu(fc_a( net, 128 ))
            action_output = fc_a( net, 4, activation=tf.nn.tanh,
                                  kernel_initializer=tf.initializers.random_uniform(minval=-0.0003,
                                                                                    maxval=0.0003))
            #输出(1,4)

            return action_output, f_s_predict

    def build_critic(self, f_s, a, scope, trainable):
        relu = partial(tf.nn.relu)
        fc_c = partial(tf.layers.dense, activation=None, trainable=trainable)
        with tf.variable_scope(scope):

            net = tf.concat([f_s, a], axis=1)
            net = relu(fc_c( net, 128 ))
            net = relu(fc_c( net, 128 ))

            q = fc_c(net, 1, kernel_initializer=tf.initializers.random_uniform(minval=-0.0003,
                                                                               maxval=0.0003))
            # Q(s,a) 输出一个[None,1]
            return q
