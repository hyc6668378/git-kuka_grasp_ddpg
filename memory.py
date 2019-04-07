# coding=utf-8
import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


def array_min2d(x):
    x = np.array(x, dtype=np.float32)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    """
    Memory 父类.

    ---- 主要类方法 ----------|-------------- 描 述 -----------------|
          append()                  添加经验（transition）
    _get_batches_for_idxes()        基于优先顺序得到采样索引
          sample()                  基于采样索引, 采样.
    """
    def __init__(self, capacity):
        self.storage = []  # Actual underlying data structure.
        self.capacity = capacity  # Maximal size.
        self._next_idx = 0  # Pointer to storage
        self.adding_demonstrations = False  # Whether the initial period of collecting demonstrations is done.
        self.num_demonstrations = 0  # Number of demo transitions in the buffer.
        self._total_transitions = 0  # Total number of collected transitions (should grow linearly)
        self.storable_elements = [
             "obs0", "actions", "rewards", "obs1", "full_states0", "full_states1", "terminals1"
        ]  # List of entities we want to store for each timestep.

    def __len__(self):
        return len(self.storage)

    @property
    def total_transitions(self):
        return self._total_transitions

    @property
    def nb_entries(self):
        return len(self.storage)

    def append(self, full_state0, obs0, action, reward, full_state1, obs1, terminal1, training=True, count=True):
        # Demonstrations are not counted towards the _total_transitions.
        if count:
            self._total_transitions += 1

        if not training:
            return False
        entry = {'obs0': obs0,
                'actions': action,
                'rewards': reward,
                'obs1': obs1,
                'full_states0': full_state0,
                'full_states1': full_state1,
                'terminals1': terminal1}

        assert len(entry) == len(self.storable_elements)

        if self._next_idx >= len(self.storage):
            self.storage.append(entry) # new transition
        else:
            self.storage[self._next_idx] = entry # cover the old transition
        self._next_idx = int(self._next_idx + 1)

        if self._next_idx >= self.capacity:
            self._next_idx = self.num_demonstrations

        return True

    def append_demonstration(self, *args, **kwargs):
        assert len(args) == len(self.storable_elements)
        assert self.adding_demonstrations
        if not self.append(*args, count=False, **kwargs):
            return
        self.num_demonstrations += 1

    def _get_batches_for_idxes(self, idxes):
        # Create a dictionary with a list for each storable entity.
        batches = {
            storable_element: [] for storable_element in self.storable_elements
        }
        for i in idxes:
            entry = self.storage[i]
            # Sanity check the size of entries
            assert len(entry) == len(self.storable_elements)
            for j, element_class in enumerate(entry):
                batches[self.storable_elements[j]].append(entry[element_class])
        result = {k: array_min2d(v) for k, v in batches.items()}
        return result

    def sample(self, batch_size):
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
        self._max_priority = 1.0

    def prior_append(self, full_state0, obs0, action, reward, full_state1, obs1, terminal1):
        idx = self._next_idx
        if not super().append(full_state0, obs0, action, reward, full_state1, obs1, terminal1):
            return
        # 新加入的'transition'设置为最大优先级, 确保每一个'transition'都至少被采样一次
        self._it_sum[idx] = self._max_priority
        self._it_min[idx] = self._max_priority

    def append_demonstration(self, full_state0, obs0, action, reward, full_state1, obs1, terminal1):
        idx = self._next_idx
        if not super().append(full_state0, obs0, action, reward, full_state1, obs1, terminal1, count=False):
            return
        self._it_sum[idx] = self._max_priority
        self._it_min[idx] = self._max_priority
        self.num_demonstrations += 1

    def _sample_proportional(self, batch_size, pretrain):
        """
        利用sum-tree快速根据优先级采样
        :return: 返回采样索引res
        """
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

    def sample_prioritized(self, batch_size, beta, pretrain=False):
        idxes = self._sample_proportional(batch_size, pretrain)
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
        encoded_sample['idxes'] = idxes
        encoded_sample['demos'] = array_min2d(demos)
        return encoded_sample

    def sample_rollout(self, batch_size, nsteps, beta, gamma, pretrain=False):
        batches = self.sample_prioritized(batch_size, beta, pretrain)
        n_step_batches = {
            storable_element: []
            for storable_element in self.storable_elements
        }
        n_step_batches["step_reached"] = []
        idxes = batches["idxes"]
        for idx in idxes:
            local_idxes = list(range(idx, min(idx + nsteps, len(self))))
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
            n_step_batches["terminals1"].append(terminal)
            n_step_batches["rewards"].append(summed_reward)
            n_step_batches["full_states1"].append(transitions["full_states1"][count])
            n_step_batches["actions"].append(transitions["actions"][0])
        n_step_batches = {
            k: array_min2d(v)
            for k, v in n_step_batches.items()
        }
        n_step_batches['idxes'] = idxes
        n_step_batches['weights'] = batches['weights']

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
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha
            self._max_priority = max(self._max_priority, priority**self._alpha)
