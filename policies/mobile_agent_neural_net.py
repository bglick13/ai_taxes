import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from policies.base_neural_net import BaseNeuralNet


def cnn_output_shape_calculator(in_dim, kernel_size):
    return np.floor(in_dim - (kernel_size - 1) - 1 + 1)


class MobileAgentNeuralNet(BaseNeuralNet):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.lstm_size = 128
        self.dnn_size = 128