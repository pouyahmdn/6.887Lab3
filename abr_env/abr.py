import numpy as np
from collections import deque

from abr_env import box, discrete, logger
from abr_env.trace_master import TraceMasterGenerative


class ABRSimEnv(object):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id=2787486
    """
    MAX_BUFFER_S = 40

    def __init__(self, seed=42):
        # setup logging
        logger.set_logging()
        # observation and action space
        self.setup_space()
        # load all trace files
        self.trace_master = TraceMasterGenerative(seed)
        # set up seed
        self.seed(seed)
        # mapping between action and bitrate level
        self.bitrate_map = [0.3, 0.75, 1.2, 1.85, 2.85, 4.3]  # Mbps
        # how many past throughput to report
        self.past_chunk_len = 10
        self.total_num_chunks = self.trace_master.get_chunk_len()

        self.trace_cap = None
        self.trace_time = None
        self.curr_t_idx = None
        self.chunk_sizes = None
        self.chunk_time_left = None
        self.chunk_idx = None
        self.buffer_size = None
        self.past_action = None
        self.past_chunk_throughputs = None
        self.past_chunk_download_times = None
        self.reset()

    def observe(self):
        if self.chunk_idx < self.total_num_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        # network throughput of past chunk, past chunk download time,
        # current buffer, number of chunks left and the last bitrate choice
        obs_arr = [self.past_chunk_throughputs[-i] for i in range(1, 9)]
        obs_arr.extend([self.past_chunk_download_times[-i] for i in range(1, 9)])
        obs_arr.extend([self.buffer_size, self.total_num_chunks - self.chunk_idx, valid_past_action])

        # current chunk size of different bitrates
        obs_arr.extend(self.chunk_sizes[i][valid_chunk_idx] for i in range(6))

        # fit in observation space
        for i in range(len(obs_arr)):
            if obs_arr[i] > self.obs_high[i]:
                logger.warn('Observation at index ' + str(i) +
                            ' at chunk index ' + str(self.chunk_idx) +
                            ' has value ' + str(obs_arr[i]) +
                            ', which is larger than obs_high ' +
                            str(self.obs_high[i]))
                obs_arr[i] = self.obs_high[i]

        obs_arr = np.array(obs_arr)
        assert self.observation_space.contains(obs_arr), obs_arr

        return obs_arr

    def reset(self, trace_index=None):
        if trace_index is None:
            self.trace_cap, self.trace_time, self.curr_t_idx, self.chunk_sizes = self.trace_master.sample_trace()
        else:
            self.trace_cap, self.trace_time, self.curr_t_idx, self.chunk_sizes = \
                self.trace_master.get_trace(trace_index)
        self.chunk_time_left = self.trace_time[self.curr_t_idx]
        self.chunk_idx = 0
        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=self.past_chunk_len)
        self.past_chunk_download_times = deque(maxlen=self.past_chunk_len)
        for _ in range(self.past_chunk_len):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)

        return self.observe()

    def seed(self, seed):
        self.trace_master.seed(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * (16) + [0, 0, 0] + [0] * 6)
        self.obs_high = np.array([20e6] * 8 + [100] * 8 + [100, 500, 5] + [10e6] * 6)
        self.observation_space = box.Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = discrete.Discrete(6)

    def step(self, action):
        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.chunk_sizes[action][self.chunk_idx]

        # compute chunk download time based on trace
        delay = 0  # in seconds
        thr_per_second_samples = []

        # keep experiencing the network trace
        # until the chunk is downloaded
        while chunk_size > 1e-8:  # floating number business

            throughput = self.trace_cap[self.curr_t_idx] / 8.0 * 1e6  # bytes/second
            chunk_time_used = min(self.chunk_time_left, chunk_size / throughput)

            chunk_size -= throughput * chunk_time_used
            self.chunk_time_left -= chunk_time_used
            delay += chunk_time_used

            if self.chunk_time_left == 0:
                thr_per_second_samples.append(self.trace_cap[self.curr_t_idx])
                self.curr_t_idx += 1
                if self.curr_t_idx == len(self.trace_time):
                    self.curr_t_idx = 0

                self.chunk_time_left = self.trace_time[self.curr_t_idx]

        # compute buffer size
        rebuffer_time = max(delay - self.buffer_size, 0)

        # update video buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += 4.0  # each chunk is 4 seconds of video

        # cap the buffer size
        self.buffer_size = min(self.buffer_size, self.MAX_BUFFER_S)

        # bitrate change penalty
        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(self.bitrate_map[action] - self.bitrate_map[self.past_action])

        # linear reward
        # (https://dl.acm.org/citation.cfm?id=3098843 section 5.1, QoE metrics (1))
        reward = self.bitrate_map[action] - 4.3 * rebuffer_time - bitrate_change

        # store action for future bitrate change penalty
        self.past_action = action

        # update observed network bandwidth and duration
        self.past_chunk_throughputs.append(
            self.chunk_sizes[action][self.chunk_idx] / float(delay))
        self.past_chunk_download_times.append(delay)

        # advance video
        self.chunk_idx += 1
        done = (self.chunk_idx == self.total_num_chunks)

        return self.observe(), reward, done, {
            'bitrate': self.bitrate_map[action],
            'stall_time': rebuffer_time,
            'bitrate_change': bitrate_change,
            'delay': delay,
            'time_sample_thrs': thr_per_second_samples
        }
