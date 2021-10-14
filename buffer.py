import numpy as np


class TransitionBuffer(object):
    def __init__(self, obs_len, size):
        """

        :param obs_len:
        :param size:
        :type obs_len: int
        :type size: int
        """
        self.cap = size
        self.obs_len = obs_len

        self.states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.next_states_buffer = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.actions_buffer = np.zeros([self.cap, 1], dtype=np.int64)
        self.rewards_buffer = np.zeros([self.cap], dtype=np.float32)
        self.dones_buffer = np.zeros([self.cap], dtype=np.bool)
        self.action_fifo = np.zeros([self.cap], dtype=np.float32)
        self.states_fifo = np.zeros([self.cap, self.obs_len], dtype=np.float32)
        self.stall_fifo = np.zeros([self.cap], dtype=np.float32)
        self.reward_fifo = np.zeros([self.cap], dtype=np.float32)

        # buffer head
        self.num_samples_so_far = 0
        self.samples_left_to_epoch = self.cap
        self.b = 0

    def reset_head(self):
        self.samples_left_to_epoch = self.cap

    def get(self):
        """

        :return:
        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        return self.states_buffer, \
            self.next_states_buffer, \
            self.actions_buffer, \
            self.rewards_buffer, \
            self.dones_buffer

    def get_all(self):
        """

        :return:
        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        return self.states_buffer, \
            self.next_states_buffer, \
            self.actions_buffer, \
            self.rewards_buffer, \
            self.dones_buffer, \
            self.stall_fifo, \
            self.reward_fifo, \
            self.action_fifo, \
            self.states_fifo

    def _place_exp(self, state, action, reward, next_state, done, index):
        """

        :type state: np.ndarray
        :type action: int
        :type reward: float
        :type next_state: np.ndarray
        :type done: bool
        :type index: int
        """
        self.states_buffer[index, :] = state
        self.next_states_buffer[index, :] = next_state
        self.actions_buffer[index, 0] = action
        self.rewards_buffer[index] = reward
        self.dones_buffer[index] = done

    def _place_batch_exp(self, states, actions, rewards, next_states, dones, index_start):
        """

        :type states: np.ndarray
        :type actions: np.ndarray
        :type rewards: np.ndarray
        :type next_states: np.ndarray
        :type dones: np.ndarray
        :type index_start: int
        """
        b_len = len(rewards)
        self.states_buffer[index_start:index_start + b_len, :] = states
        self.next_states_buffer[index_start: index_start + b_len, :] = next_states
        self.actions_buffer[index_start: index_start + b_len] = actions
        self.rewards_buffer[index_start: index_start + b_len] = rewards
        self.dones_buffer[index_start: index_start + b_len] = dones

    def add_exp(self, state, action, reward, next_state, done, stall, state_fifo_entry=None,
                action_fifo_entry=None, stall_fifo_entry=None, reward_fifo_entry=None):
        """

        :type state: np.ndarray
        :type state_fifo_entry: np.ndarray
        :type action: int
        :type action_fifo_entry: int
        :type reward: float
        :type reward_fifo_entry: float
        :type next_state: np.ndarray
        :type done: bool
        :type stall: float
        :type stall_fifo_entry: float
        """
        self.action_fifo[self.cap - self.samples_left_to_epoch] = \
            action if action_fifo_entry is None else action_fifo_entry
        self.reward_fifo[self.cap - self.samples_left_to_epoch] = \
            reward if reward_fifo_entry is None else reward_fifo_entry
        self.stall_fifo[self.cap - self.samples_left_to_epoch] = \
            stall if stall_fifo_entry is None else stall_fifo_entry
        self.states_fifo[self.cap - self.samples_left_to_epoch, :] = \
            state if state_fifo_entry is None else state_fifo_entry
        self._place_exp(state, action, reward, next_state, done, self.b)
        self.b += 1
        self.samples_left_to_epoch -= 1
        self.num_samples_so_far += 1
        if self.b == self.cap:
            self.b = 0
        assert self.b <= self.cap

    def batch_add_exp(self, states, actions, rewards, next_states, dones,
                      states_fifo, actions_fifo, rewards_fifo, stalls_fifo):
        """

        :type states: np.ndarray
        :type actions: np.ndarray
        :type rewards: np.ndarray
        :type next_states: np.ndarray
        :type dones: np.ndarray
        :type stalls_fifo: np.ndarray
        :type states_fifo: np.ndarray
        :type actions_fifo: np.ndarray
        :type rewards_fifo: np.ndarray
        """
        b_len = len(rewards)
        self.action_fifo[self.cap - self.samples_left_to_epoch:
                         self.cap - self.samples_left_to_epoch + b_len] = actions_fifo
        self.reward_fifo[self.cap - self.samples_left_to_epoch: self.cap - self.samples_left_to_epoch + b_len] = rewards_fifo
        self.stall_fifo[self.cap - self.samples_left_to_epoch: self.cap - self.samples_left_to_epoch + b_len] = stalls_fifo
        self.states_fifo[self.cap - self.samples_left_to_epoch: self.cap - self.samples_left_to_epoch + b_len, :] = states_fifo
        self._place_batch_exp(states, actions, rewards, next_states, dones, self.b)
        self.b += b_len
        self.samples_left_to_epoch -= b_len
        self.num_samples_so_far += b_len
        if self.b == self.cap:
            self.b = 0
        assert self.b <= self.cap

    def buffer_full(self):
        """

        :return:
        :rtype: bool
        """
        return self.samples_left_to_epoch <= 0
