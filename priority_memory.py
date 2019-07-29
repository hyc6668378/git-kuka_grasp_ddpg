# coding=utf-8

import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

# 加入demo 时候也要注意 数据类型
def array_min2d(x):
    x = np.array(x, dtype=np.float32)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, capacity):
        self.storage = []  # Actual underlying data structure.
        self.capacity = capacity  # Maximal size.
        self._next_idx = 0  # Pointer to storage
        self.adding_demonstrations = True  # Whether the initial period of collecting demonstrations is done.
        self.num_demonstrations = 0  # Number of demo transitions in the buffer.
        self._total_transitions = 0  # Total number of collected transitions (should grow linearly)
        self.storable_elements = [
             "obs0",
             "obs1",
             "f_s0",
             "f_s1",
             "actions",
             "rewards",
             "terminals1"
        ]  # List of entities we want to store for each timestep.

    def __len__(self):
        return len(self.storage)

    @property
    def total_transitions(self):
        return self._total_transitions

    @property
    def nb_entries(self):
        return len(self.storage)

    def append_(self, obs0,
               obs1,
               f_s0,
               f_s1,
               actions,
               rewards,
               terminal1,
               training=True,
               count=True):
        # Demonstrations are not counted towards the _total_transitions.
        if count:
            self._total_transitions += 1

        if not training:
            return False
        entry = {
                 'obs0': obs0,
                 'obs1': obs1,
                 'f_s0': f_s0,
                 'f_s1': f_s1,
                 'actions': actions,
                 'rewards': rewards,
                 'terminals1': terminal1,
        }

        assert len(entry) == len(self.storable_elements)

        if self._next_idx >= len(self.storage):
            self.storage.append(entry) # new transition
        else:
            self.storage[self._next_idx] = entry # cover the old transition
        self._next_idx = int(self._next_idx + 1)
        if self._next_idx >= self.capacity:
            self._next_idx = self.num_demonstrations # TODO: 从头覆盖会影响到 n-step return 的终点判断. 需要优化
        return True

    def _get_batches_for_idxes(self, idxes):
        batches = {
            storable_element: [] for storable_element in self.storable_elements
        }
        for i in idxes:
            entry = self.storage[i]
            assert len(entry) == len(self.storable_elements)
            for j, elem_name in enumerate(entry):
                batches[elem_name].append(entry[elem_name])
        result = {k: array_min2d(v) for k, v in batches.items()}
        return result

    def sample_(self, batch_size):
        idxes = np.random.random_integers(
            low=0, high=self.nb_entries - 1, size=batch_size)
        demos = [i < self.num_demonstrations for i in idxes]
        encoded_sample = self._get_batches_for_idxes(idxes)
        encoded_sample['weights'] = array_min2d(np.ones((batch_size, )))
        encoded_sample['idxes'] = idxes
        encoded_sample['demos'] = array_min2d(demos)
        return encoded_sample

    def demonstrations_done(self):
        self.adding_demonstrations = False


class PrioritizedMemory(Memory):
    def __init__(self,
                 capacity,
                 alpha,
                 transition_small_epsilon=1e-6,
                 demo_epsilon=0.2):
        super(PrioritizedMemory, self).__init__(capacity)
        assert alpha > 0
        self._alpha = alpha
        self._transition_small_epsilon = transition_small_epsilon
        self._demo_epsilon = demo_epsilon
        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2  # Size must be power of 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 100.0
        self.help = []

    def append(self, obs0,
               obs1,
               f_s0,
               f_s1,
               action,
               reward,
               terminal1
               ):
        idx = self._next_idx
        if not super(PrioritizedMemory, self).append_(obs0=obs0,
                                                      obs1=obs1,
                                                      f_s0=f_s0,
                                                      f_s1=f_s1,
                                                      actions=action,
                                                      rewards=reward,
                                                      terminal1=terminal1):
            return
        # 新加入的'transition'设置为最大优先级, 确保每一个'transition'都至少被采样一次
        self._it_sum[idx] = self._max_priority
        self._it_min[idx] = self._max_priority

    def append_demo(self, obs0, obs1, f_s0, f_s1,
                    action, reward, terminal1):
        idx = self._next_idx
        if not super(PrioritizedMemory, self).append_(obs0=obs0,
                                                      obs1=obs1,
                                                      f_s0=f_s0,
                                                      f_s1=f_s1,
                                                      actions=action,
                                                      rewards=reward,
                                                      terminal1=terminal1,
                                                      count=False):
            return
        self._it_sum[idx] = self._max_priority
        self._it_min[idx] = self._max_priority
        self.num_demonstrations += 1

    def _sample_proportional(self, batch_size, pretrain):
        res = []
        if pretrain:
            res = np.random.random_integers(
                low=0, high=self.nb_entries - 1, size=batch_size)
            return res
        for _ in range(batch_size):
            while True:
                mass = np.random.uniform(
                    0, self._it_sum.sum(0, len(self.storage) - 1))
                idx = self._it_sum.find_prefixsum_idx(mass)
                if idx not in res:
                    res.append(idx)
                    break
        return res

    def sample(self, batch_size, beta, pretrain=False):
        idxes = self._sample_proportional(batch_size, pretrain)
        # demos is a bool
        demos = [i < self.num_demonstrations for i in idxes]
        weights = []
        p_sum = self._it_sum.sum()
        # 算重要性采样权重 weights
        for idx in idxes:
            p_sample = self._it_sum[idx] / p_sum
            weight = ((1.0 / p_sample) * (1.0 / len(self.storage)))**beta
            weights.append(weight)
        weights = np.array(weights) / np.max(weights)
        encoded_sample = self._get_batches_for_idxes(idxes)
        encoded_sample['weights'] = array_min2d(weights)
        encoded_sample['idxes'] = array_min2d(idxes)
        encoded_sample['demos'] = array_min2d(demos)
        return encoded_sample


    def sample_rollout(self, batch_size, nsteps, beta, gamma, pretrain=False):
        batches = self.sample(batch_size, beta, pretrain)
        n_step_batches = {
            storable_element: []
            for storable_element in self.storable_elements
        }
        n_step_batches["step_reached"] = []
        idxes = batches["idxes"]
        for idx in idxes:
            local_idxes = list(range(int(idx), int(min(idx + nsteps, len(self)))))
            transitions = self._get_batches_for_idxes(local_idxes)
            summed_reward = 0
            count = 0
            terminal = 0.0
            terminals = transitions['terminals1']
            r = transitions['rewards']
            for i in range(len(r)):
                summed_reward += (gamma**i) * r[i][0]
                count = i
                if terminals[i]:
                    terminal = 1.0
                    break

            n_step_batches["step_reached"].append(count)
            n_step_batches["obs1"].append(transitions["obs1"][count])
            n_step_batches["f_s1"].append(transitions["f_s1"][count])
            n_step_batches["terminals1"].append(terminal)
            n_step_batches["rewards"].append(summed_reward)
            n_step_batches["actions"].append(transitions["actions"][0])
            n_step_batches['demos'] = batches['demos']
        n_step_batches['weights'] = batches['weights']
        n_step_batches['idxes'] = idxes

        n_step_batches = {
            k: array_min2d(v)
            for k, v in n_step_batches.items()
        }

        return batches, n_step_batches, sum(batches['demos']) / batch_size

    def update_priorities(self, idxes, td_errors, actor_losses=0.0):
        priorities = td_errors + \
            (actor_losses ** 2) + self._transition_small_epsilon
        for i in range(len(priorities)):
            if idxes[i] < self.num_demonstrations:
                priorities[i] += np.max(priorities) * self._demo_epsilon
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.storage)
            idx = int(idx)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha
            self._max_priority = max(self._max_priority, priority**self._alpha)
            self.help.append(priority**self._alpha)