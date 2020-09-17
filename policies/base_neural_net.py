import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from util.observation_batch import ObservationBatch


def cnn_output_shape_calculator(in_dim, kernel_size):
    return np.floor(in_dim - (kernel_size - 1) - 1 + 1)


class BaseNeuralNet(nn.Module):
    def build_models(self, obs):
        self.obs = obs
        self._build_cnn()
        self._build_dnn()
        self._build_temporal_model()
        self._build_action_head()
        self._build_value_head()
        self.obs = None

    def _build_cnn(self):
        layers = [nn.Conv2d(self.obs.world_map.shape[1], 32, 3),
                   nn.ReLU(),
                   nn.Conv2d(32, 64, 2),
                   nn.ReLU()]
        self.cnn = nn.Sequential(*layers).to(self.device)
        self.cnn_output_shape = self.cnn(torch.FloatTensor(self.obs.world_map).to(self.device)).shape

    def _build_dnn(self):
        layers = [nn.Linear(int(np.array(self.cnn_output_shape[1:]).prod()) + self.obs.flat_inputs.shape[1], self.dnn_size),
                  nn.ReLU(),
                  nn.Linear(self.dnn_size, self.dnn_size),
                  nn.ReLU()]
        self.dnn = nn.Sequential(*layers).to(self.device)

    def _build_temporal_model(self):
        self.temporal_model = nn.LSTM(self.lstm_size, self.lstm_size, 1).to(self.device)

    def _build_action_head(self):
        self.action_head = nn.Sequential(*[nn.Linear(self.lstm_size, self.obs['action_mask'].shape[1])]).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, self.obs['action_mask'].shape[1])).to(self.device)

    def _build_value_head(self):
        self.value_head = nn.Sequential(*[nn.Linear(self.lstm_size, 1)]).to(self.device)

    def _batchify(self):
        if isinstance(self.obs, dict):
            self.obs = pd.DataFrame.from_dict(self.obs, orient='index').to_dict()  # Basically just transpose the dict

    def _get_local_map(self):
        h = self.cnn(torch.FloatTensor(self.obs.world_map).to(self.device))
        return h

    def _concat_state_space(self, h):
        local_map = h.reshape(h.shape[0], -1)
        other_obs = self.obs.flat_inputs
        try:
            other_obs = torch.FloatTensor(other_obs).to(self.device)
        except TypeError:
            print(other_obs)
        h = torch.cat((local_map, other_obs), 1).to(self.device)
        return h

    def _process_concat_state_space(self, h):
        return self.dnn(h)

    def _update_temporal_model(self, h, prev_hc):
        if h.shape[0] != prev_hc[0].shape[1]:
            prev_hc = tuple(phc.repeat(1, h.shape[0], 1) for phc in prev_hc)
        _, hc = self.temporal_model(h.unsqueeze(0), prev_hc)
        return hc

    def _compute_actions(self, h, det=False):
        if det:
            return self.action_head(h).argmax(-1)
        mu = self.action_head(h)
        try:
            action_mask = torch.FloatTensor(self.obs.action_mask).to(self.device)
            mu *= action_mask
        except Exception as e:
            print(f'exception in action mask:\n{e}')
        mu = mu.softmax(-1)
        dist = torch.distributions.Categorical(mu)
        return dist

    def _compute_value(self, h):
        return self.value_head(h)

    def __init__(self, device='cuda'):
        super().__init__()
        self.lstm_size = None
        self.dnn_size = None

        self.cnn = None
        self.dnn = None
        self.temporal_model = None
        self.action_head = None
        self.value_head = None
        self.log_std = None

        self.obs = None
        self.batch_size = None

        self.device = device

    def forward(self, obs: ObservationBatch, prev_hc=None, det=False):
        self.obs = obs
        h = self._get_local_map()
        h = self._concat_state_space(h)
        h = self._process_concat_state_space(h)
        hc = self._update_temporal_model(h, prev_hc)
        dist = self._compute_actions(hc[0].squeeze(), det=det)
        value = self._compute_value(hc[0].squeeze())
        return dist, value, hc

    def act(self):
        return F.softmax(self.h)