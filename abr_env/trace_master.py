import pickle

import numpy as np
from abr_env.trace_loader import load_chunk_sizes


class TraceMasterGenerative(object):
    def __init__(self, seed):
        """
        :param seed:
        :type seed: int
        """
        self.seed_rng = seed
        self.rng_trace = np.random.RandomState(seed)
        self.chunk_sizes = load_chunk_sizes()
        self.len_trace = self.get_chunk_len()
        # assert number of chunks for different bitrates are all the same
        assert len(np.unique([len(chunk_size) for chunk_size in self.chunk_sizes])) == 1

        self.dict_states = [
            {'states_a': 0.2, 'states_b': 5, 'c_freq': 0.5, 'trans_sigma': 0.5, 'rw_iid': 0.3, 'rw_scale': 0.3},
        ]

        self.nc = 100
        self.num_states = len(self.dict_states)
        self.state_space = np.empty((self.num_states, self.nc))
        self.p_init_states = np.empty((self.num_states, self.nc))
        self.p_trans_states = np.empty((self.num_states, self.nc, self.nc))
        for i, state_info in enumerate(self.dict_states):
            self.state_space[i] = np.linspace(state_info['states_a'], state_info['states_b'], self.nc, endpoint=True)
            self.p_init_states[i] = np.ones(self.nc)/self.nc
            self.p_trans_states[i] = self.state_space[i][np.newaxis, ...] - self.state_space[i][..., np.newaxis]
            self.p_trans_states[i] = np.exp(-self.p_trans_states[i] ** 2 / state_info['trans_sigma'])
            self.p_trans_states[i] *= (1 - np.eye(self.nc))
            self.p_trans_states[i] /= self.p_trans_states[i].sum(axis=-1, keepdims=True)

        self.step = 0
        self.curr_state = 0
        
        trace = np.empty(self.len_trace)

        state = self.rng_trace.choice(self.nc, p=self.p_init_states[self.curr_state])
        trace[0] = self.state_space[self.curr_state, state]
        freq = self.rng_trace.beta(1, 3) * self.dict_states[self.curr_state]['c_freq']
        for k in range(1, trace.shape[0]):
            if self.rng_trace.random() > freq:
                trace[k] = trace[k - 1] + \
                           self.rng_trace.normal(-self.dict_states[self.curr_state]['rw_iid'] *
                                                 (trace[k - 1] - self.state_space[self.curr_state, state]),
                                                 np.sqrt(self.dict_states[self.curr_state]['rw_iid'] * 2) *
                                                 self.dict_states[self.curr_state]['rw_scale'])
            else:
                state = self.rng_trace.choice(self.nc, p=self.p_trans_states[self.curr_state][state])
                trace[k] = self.state_space[self.curr_state, state]
            trace[k] = max(trace[k], 0.2)
        time_trace = self.rng_trace.random(self.len_trace) * 0.4 + 0.8
        
        self.trace_single = trace
        self.time_single = time_trace

    def seed(self, seed):
        """

        :param seed:
        :type seed: int
        """
        self.seed_rng = seed
        self.rng_trace = np.random.RandomState(seed)

    def sample_trace(self):
        """

        :rtype: (np.ndarray, np.ndarray, int, np.ndarray)
        """
        
        trace = np.empty(self.len_trace)

        state = self.rng_trace.choice(self.nc, p=self.p_init_states[self.curr_state])
        trace[0] = self.state_space[self.curr_state, state]
        freq = self.rng_trace.beta(1, 3) * self.dict_states[self.curr_state]['c_freq']
        for k in range(1, trace.shape[0]):
            if self.rng_trace.random() > freq:
                trace[k] = trace[k - 1] + \
                           self.rng_trace.normal(-self.dict_states[self.curr_state]['rw_iid'] *
                                                 (trace[k - 1] - self.state_space[self.curr_state, state]),
                                                 np.sqrt(self.dict_states[self.curr_state]['rw_iid'] * 2) *
                                                 self.dict_states[self.curr_state]['rw_scale'])
            else:
                state = self.rng_trace.choice(self.nc, p=self.p_trans_states[self.curr_state][state])
                trace[k] = self.state_space[self.curr_state, state]
            trace[k] = max(trace[k], 0.2)
        time_trace = self.rng_trace.random(self.len_trace) * 0.4 + 0.8
        
        self.trace_single = trace
        self.time_single = time_trace
        
        return self.trace_single, self.time_single, 0, self.chunk_sizes

    def get_count(self):
        """

        :return:
        :rtype: int
        """
        return 1000

    def get_trace(self, index):
        """
        :param index:
        :type index: int

        :rtype: (np.ndarray, np.ndarray, int, np.ndarray)
        """
        return self.sample_trace()

    def get_curr_trace(self):
        """
        :rtype: int
        """
        return self.curr_state

    def set_curr_trace(self, state):
        """
        :param state:
        :type state: int
        """
        self.curr_state = state

    def get_chunk_len(self):
        """
        :rtype: int
        """
        return self.chunk_sizes.shape[1]
