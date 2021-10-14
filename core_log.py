import numpy as np

from buffer import TransitionBuffer
from torch.utils.tensorboard import SummaryWriter


def log_a2c(buff, ret_np, v_np, adv_np, pg_loss, v_loss, entropy_factor, norm_entropy, elapsed, monitor, epoch):
    """
    :type buff: TransitionBuffer
    :type ret_np: np.ndarray
    :type v_np: np.ndarray
    :type adv_np: np.ndarray
    :type pg_loss: float
    :type v_loss: float
    :type entropy_factor: float
    :type norm_entropy: float
    :type elapsed: float
    :type monitor: SummaryWriter
    :type epoch: int
    """
    avg_reward = np.mean(buff.reward_fifo)
    avg_action = np.mean(buff.action_fifo)
    avg_stall = np.mean(buff.stall_fifo)
    buffer_secs = buff.states_fifo[:, 16]

    monitor.add_scalar('Buffer/min', np.min(buffer_secs), epoch)
    monitor.add_scalar('Buffer/avg', np.mean(buffer_secs), epoch)
    monitor.add_scalar('Buffer/max', np.max(buffer_secs), epoch)
    
    monitor.add_scalar('Policy/avg_reward', avg_reward, epoch)
    monitor.add_scalar('Policy/avg_action', avg_action, epoch)
    monitor.add_scalar('Policy/avg_stall', avg_stall, epoch)
    
    monitor.add_scalar('Time/elapsed', elapsed, epoch)
    
    monitor.add_histogram('Action/actions', buff.action_fifo, epoch)

    dim_mean_obs = buff.states_fifo.mean(axis=tuple(range(buff.states_fifo.ndim - 1)))
    for i in range(dim_mean_obs.shape[0]):
        monitor.add_scalar('Obs/dim%d' % i, dim_mean_obs[i], epoch)

    # gather statistics
    ret_mean = ret_np.mean()
    v_net_mean = v_np.mean()
    adv_mean = adv_np.mean()

    # monitor statistics
    monitor.add_scalar('Loss/pg_loss', pg_loss, epoch)
    monitor.add_scalar('Loss/v_loss', v_loss, epoch)
    monitor.add_scalar('Loss/value_target', ret_mean, epoch)
    monitor.add_scalar('Loss/adv_mean', adv_mean, epoch)
    monitor.add_scalar('Loss/value_network', v_net_mean, epoch)
    
    monitor.add_scalar('Policy/entropy_factor', entropy_factor, epoch)
    monitor.add_scalar('Policy/norm_entropy', norm_entropy, epoch)