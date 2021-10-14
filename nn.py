import torch
import torch.nn as nn
import torch.nn.functional as tfunctional


class PensNet(nn.Module):
    def __init__(self, obs_width, out_space, nn_hid):
        super(PensNet, self).__init__()
        self.obs_width = obs_width
        self.out_space = out_space
        modules = []
        input_list = [self.obs_width] + nn_hid
        output_list = nn_hid + [self.out_space]
        for i in range(len(input_list)):
            modules.append(nn.Linear(input_list[i], output_list[i]))
            if i != len(input_list) - 1:
                modules.append(nn.ReLU())
        self.mapper = nn.Sequential(*modules)

    def forward(self, observation):
        return self.mapper(observation).squeeze(dim=-1)
    
    @torch.jit.export
    def sample_policy(self, observation):
        """
        :type observation: torch.Tensor
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            pi = self.forward(observation)
            pi_cpu = tfunctional.softmax(pi, dim=-1).cpu()
        return pi_cpu