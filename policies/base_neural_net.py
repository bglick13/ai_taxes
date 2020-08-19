import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def cnn_output_shape_calculator(in_dim, kernel_size):
    return np.floor(in_dim - (kernel_size - 1) - 1 + 1)


class BaseNeuralNet(nn.Module):
    def build_models(self, obs):
        self.obs = obs
        self._build_cnn()
        self._build_dnn()
        self._build_temporal_model()
        self._build_action_head()
        self.obs = None

    def _build_cnn(self):
        layers = [nn.Conv2d(self.obs['world-map'].shape[0], 32, 3),
                   nn.ReLU(),
                   nn.Conv2d(32, 64, 2),
                   nn.ReLU()]
        self.cnn = nn.Sequential(*layers)
        self.cnn_output_shape = self.cnn(self.obs['world-map']).shape

    def _build_dnn(self):
        layers = [nn.Linear(self.cnn_output_shape.view(-1, 1)[1], self.dnn),
                  nn.ReLU,
                  nn.Linear(self.dnn, self.dnn),
                  nn.ReLU]
        self.dnn = nn.Sequential(*layers)

    def _build_temporal_model(self):
        self.temporal_model = nn.LSTM(self.lstm_size, self.lstm_size, 1)
        self.hidden_cell = None

    def _build_action_head(self):
        self.action_head = nn.Linear(self.lstm_size, self.obs['action_mask'].shape)

    def _get_local_map(self):
        self.h = self.cnn(self.obs['world-map'])

    def _concat_state_space(self):
        local_map = self.h.reshape(1, -1)
        other_obs = []
        for key, value in self.obs.items():
            if isinstance(value, float) or len(np.array(value).shape) == 1:
                other_obs += value

        other_obs = torch.FloatTensor(other_obs)
        self.h = torch.cat((local_map, other_obs))

    def _process_concat_state_space(self):
        self.h = self.dnn(self.h)

    def _update_temporal_model(self):
        self.h = self.temporal_model(self.h)

    def _compute_actions(self):
        self.h = self.action_head(self.h)

    def __init__(self):
        super().__init__()
        self.lstm_size = None
        self.dnn_size = None

        self.cnn = None
        self.dnn = None
        self.temporal_model = None
        self.action_head = None
        self.obs = None
        self.h = None

    def forward(self, obs):
        self.obs = obs
        self._get_local_map()
        self._concat_state_space()
        self._process_concat_state_space()
        self._update_temporal_model()
        self._compute_actions()

    def act(self):
        return F.softmax(self.h)