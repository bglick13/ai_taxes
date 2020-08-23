import torch
from torch import nn
from policies.base_neural_net import BaseNeuralNet


class TaxPlannerNeuralNet(BaseNeuralNet):
    def _build_action_head(self):
        self.action_head = nn.ModuleList([nn.Linear(self.lstm_size, self.obs['action_mask'].shape // 7) for _ in range(7)]).to(self.device)

    def _compute_actions(self):
        self.h = torch.FloatTensor([ah(self.h) for ah in self.action_head]).to(self.device)

    def __init__(self, device='cuda'):
        super().__init__(device)
        self.lstm_size = 256
        self.dnn_size = 256