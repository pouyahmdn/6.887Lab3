from sklearn.preprocessing import Normalizer
import pickle
from buffer import TransitionBuffer


def create_normer():
    transformer = Normalizer()
    np.random.seed(123)
    env = ABRSimEnv(123)

    obs_len = env.observation_space.shape[0]
    act_len = env.action_space.n
    buff = TransitionBuffer(obs_len, env.total_num_chunks*100)

    buff.reset_head()
    done = True

    while not buff.buffer_full():
        if done:
            obs = env.reset()

        act = np.random.choice(act_len)

        next_obs, rew, done, info = env.step(act)

        buff.add_exp(obs, act, rew, next_obs, done, info['stall_time'])

        # next state
        obs = next_obs

    assert buff.buffer_full()

    all_states, all_next_states, all_actions_np, all_rewards, all_dones = buff.get()

    transformer.fit_transform(all_states)

    with open('normer.pkl', 'wb') as fandle:
        pickle.dump(transformer, fandle)


class Normer(object):
    def __init__(self):
        with open('normer.pkl', 'rb') as fandle:
            self.transformer = pickle.load(fandle)
    
    def __call__(self, data):
        return self._normit(data)
        
    def _normit(self, data):
        if data.ndim == 1:
            resqueeze = True
            data = data.reshape(1, -1)
        else:
            resqueeze = False
        out = self.transformer.transform(data)
        if resqueeze:
            out = out[0]
        return out
