import torch
from typing import Union, List, Dict, Tuple
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class ObservationBatch:
    def __init__(self, obs: Union[List, Dict], keys: Union[List, Tuple] = None, flatten_action_masks=True):
        self.keys = keys
        self.flatten_action_masks = flatten_action_masks
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
            if flatten_action_masks:
                action_masks = []
                for am in self.obs['action_mask'].values:
                    action_masks.append(np.concatenate(list(am.values())))
                self.action_mask = np.array(action_masks)
            self.world_map = np.array([a for a in self.obs['world-map'].values])
            columns = []
            regex_search = re.compile('p\d')
            for c in self.obs.columns:
                if '-' in c or regex_search.match(c):
                    if isinstance(self.obs[c].values[0], str):
                        continue
                    if isinstance(self.obs[c].values[0], np.ndarray):
                        if len(self.obs[c].values[0].shape) > 1:
                            continue
                    if isinstance(self.obs[c].values[0], dict):
                        for key, value in self.obs[c].values[0].items():
                            new_c = c + '-' + key
                            self.obs[new_c] = pd.Series([value])
                            columns.append(new_c)
                    else:
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