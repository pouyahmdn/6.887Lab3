import numpy as np
import torch
import torch.nn.functional as tfunctional
from torch.distributions import Categorical


def sample_action(policy_net, obs, device):
    """
    :type policy_net: torch.nn.Module
    :type obs: np.ndarray
    :type device: torch.device
    :rtype: int
    """
    pi_cpu = policy_net.sample_policy(torch.as_tensor(obs, dtype=torch.float, device=device))
    return Categorical(pi_cpu).sample().item()


def cumulative_rewards(rewards_np, dones_np, next_value_np, gamma):
    """
    :type rewards_np: np.ndarray
    :type dones_np: np.ndarray
    :type next_value_np: np.ndarray
    :type gamma: float
    :rtype: np.ndarray
    """
    returns = np.zeros(len(rewards_np), dtype=np.float32)
    last_val = next_value_np[-1]
    # if done is true (and 1), next val should be zero in return
    gamma_arr = gamma * (1-dones_np.astype(float))
    
    for i in reversed(range(len(rewards_np))):
        last_val = rewards_np[i] + gamma_arr[i] * last_val
        returns[i] = last_val

    return returns


def gae_advantage(rewards_np, dones_np, values_np, next_values_np, gamma, lam):
    """
    :type rewards_np: np.ndarray
    :type dones_np: np.ndarray
    :type values_np: np.ndarray
    :type next_values_np: np.ndarray
    :type gamma: float
    :type lam: float
    :rtype: np.ndarray
    """

    # TD lambda style advantage computation
    # more details in GAE: https://arxiv.org/pdf/1506.02438.pdf
    adv = np.zeros([len(rewards_np), 1], dtype=np.float32)
    last_gae = 0
    gamma_arr = gamma * (1-dones_np.astype(float))

    for i in reversed(range(len(rewards_np))):
        delta = rewards_np[i] + gamma_arr[i] * next_values_np[i] - values_np[i]
        adv[i] = last_gae = delta + gamma_arr[i] * lam * last_gae

    return adv


def policy_gradient(policy_net, net_opt_p, states_torch, actions_torch, adv_torch, entropy_factor):
    """
    :type policy_net: torch.nn.Module
    :type net_opt_p: torch.optim.Optimizer
    :type states_torch: torch.Tensor
    :type actions_torch: torch.Tensor
    :type adv_torch: torch.Tensor
    :type entropy_factor: float
    :rtype: float, float
    """
    q = policy_net.forward(states_torch)
    log_pi = tfunctional.log_softmax(q, dim=-1)
    log_pi_acts = log_pi.gather(1, actions_torch)

    pi = torch.exp(log_pi)
    entropy = (log_pi * pi).sum(dim=-1).mean()
    pg_loss = - (log_pi_acts * adv_torch).mean()

    loss = pg_loss + entropy_factor * entropy
    net_opt_p.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    net_opt_p.step()

    return pg_loss.item(), entropy.item()


def value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch):
    """
    :type value_net: PermInvNet | torch.nn.Module
    :type net_opt_v: torch.optim.Optimizer
    :type net_loss: torch.nn.MSELoss
    :type values_torch: torch.Tensor
    :type returns_torch: torch.Tensor
    :rtype: float
    """
    v_loss = net_loss(values_torch, returns_torch)
    net_opt_v.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1)
    net_opt_v.step()
    
    return v_loss.item()


def train_actor_critic(value_net, policy_net, net_opt_p, net_opt_v, net_loss, device,
                       actions_np, next_obs_np, rewards_np, obs_np, dones_np, entropy_factor, gamma, lam):
    """
    :type value_net: PermInvNet | torch.nn.Module
    :type policy_net: PermInvNet | torch.nn.Module
    :type net_opt_p: torch.optim.Optimizer
    :type net_opt_v: torch.optim.Optimizer
    :type net_loss: torch.nn.MSELoss
    :type device: torch.device
    :type actions_np: np.ndarray
    :type next_obs_np: np.ndarray
    :type rewards_np: np.ndarray
    :type obs_np: np.ndarray
    :type dones_np: np.ndarray
    :type entropy_factor: float
   
    :rtype: (float, float, float, np.ndarray, np.ndarray, np.ndarray)
    """

    actions_torch = torch.as_tensor(actions_np, dtype=torch.int64, device=device)
    obs_torch = torch.as_tensor(obs_np, dtype=torch.float, device=device)

    # compute values
    values_torch = value_net.forward(obs_torch)
    values_np = values_torch.cpu().detach().numpy()
    next_values_torch = value_net.forward(torch.as_tensor(next_obs_np, dtype=torch.float, device=device))
    next_values_np = next_values_torch.cpu().detach().numpy()

    # cumulative reward
    returns_np = cumulative_rewards(rewards_np, dones_np, next_values_np, gamma)
    returns_torch = torch.as_tensor(returns_np, dtype=torch.float, device=device)

    # compute advantage
    adv_np = gae_advantage(rewards_np, dones_np, values_np, next_values_np, gamma, lam)
    adv_torch = torch.as_tensor(adv_np, dtype=torch.float, device=device)

    # policy gradient training
    pg_loss, entropy = policy_gradient(policy_net, net_opt_p, obs_torch, actions_torch, adv_torch, entropy_factor)

    # value training
    v_loss = value_train(value_net, net_opt_v, net_loss, values_torch, returns_torch)

    return pg_loss, v_loss, entropy, returns_np, values_np, adv_np
