import torch
from typing import Union, List, Dict, Tuple
import numpy as np
import pandas as pd


class ObservationBatch:
    def __init__(self, obs: Union[List, Dict], keys: Union[List, Tuple] = None):
        self.keys = keys
        if isinstance(obs, list):
            self.batch_size = len(obs)
            self.world_map = obs[0]
            self.flat_inputs = obs[1]
            self.obs = obs
        elif isinstance(obs, dict):
            self.batch_size = 1
            self.obs = pd.DataFrame.from_dict(obs, orient='index').loc[keys, :]  # Basically just transpose the dict
            self.order = self.obs.index.values
            # self.obs = self.obs.to_dict()
            for key, value in self.obs.to_dict().items():
                self.__setattr__(key, np.array([_value for _key, _value in value.items()]))
            self.world_map = np.array([a for a in self.obs['world-map'].values])
            columns = []
            for c in self.obs.columns:
                if '-' in c:
                    if isinstance(self.obs[c].values[0], np.ndarray):
                        if len(self.obs[c].values[0].shape) > 1:
                            continue
                    columns.append(c)
            self.flat_inputs = self.obs.loc[:, columns].T.apply(pd.Series.explode).T.fillna(0).values

    # def world_map(self):
    #     return self.world_map
    #
    # def flat_inputs(self):
    #     return self.flat_inputs

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value