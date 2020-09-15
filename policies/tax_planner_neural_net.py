import torch
from torch import nn
from policies.base_neural_net import BaseNeuralNet
import numpy as np


class TaxPlannerNeuralNet(BaseNeuralNet):
    def _build_action_head(self):
        action_mask = self.obs['action_mask'][0]
        self.action_head = nn.ModuleList([nn.Linear(self.lstm_size, len(am)) for am in action_mask]).to(self.device)

    def _compute_actions(self, h, det=False):
        mu = torch.stack([ah(h) for ah in self.action_head]).to(self.device).permute(1, 0, 2)
        try:
            action_mask = torch.FloatTensor(np.array(list(self.obs.action_mask))).to(self.device)
            mu *= action_mask
        except:
            pass
        mu = mu.softmax(-1)
        dist = torch.distributions.Categorical(mu)
        return dist

    def __init__(self, device='cuda'):
        super().__init__(device)
        self.lstm_size = 256
        self.dnn_size = 256