import torch
from typing import Union, List, Dict
import numpy as np
import pandas as pd


class ObservationBatch:
    def __init__(self, obs: Union[List, Dict], keys: List = None):
        self.keys = keys
        if isinstance(obs, list):
            self.batch_size = len(obs)
        elif isinstance(obs, dict):
            self.batch_size = 1
            self.obs = pd.DataFrame.from_dict(obs, orient='index').loc[keys, :]  # Basically just transpose the dict
            self.order = self.obs.index.values
            # self.obs = self.obs.to_dict()
            for key, value in self.obs.to_dict().items():
                self.__setattr__(key, np.array([_value for _key, _value in value.items()]))

    @property
    def world_map(self):
        return np.array([a for a in self.obs['world-map'].values])

    @property
    def flat_inputs(self):
        columns = []
        for c in self.obs.columns:
            if '-' in c:
                if isinstance(self.obs[c].values[0], np.ndarray):
                    if len(self.obs[c].values[0].shape) > 1:
                        continue
                columns.append(c)
        return self.obs.loc[:, columns].T.apply(pd.Series.explode).T.fillna(0).values

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value