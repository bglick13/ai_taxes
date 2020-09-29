import re
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd


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
            if 'p' in keys:
                regex_search = re.compile('MalthusianPeriodicBracketTax-\w*land')
                columns = self.obs.columns.values
                nation_columns = [c for c in columns if regex_search.match(c)]
                nation_names = [c.split('-')[1] for c in nation_columns]
                # Add each nation as a row in the dataframe, similar to mobile agent obs structure
                self.obs = self.obs.reindex(['p'] + nation_names, method='pad')
                # Copy the planner obs to the nation obs
                for nn in nation_names:
                    self.obs.loc[nn, :] = self.obs.loc['p', :]
                # Combine each nation column into a single column
                tmp = self.obs.loc['p', nation_columns]
                tmp.index = nation_names
                self.obs = self.obs.drop('p', 0)
                self.obs['MalthusianPeriodicBracketTax-nation_data'] = tmp
                self.obs = self.obs.drop(nation_columns, 1)
            self.order = self.obs.index.values
            # self.obs = self.obs.to_dict()
            for key, value in self.obs.to_dict().items():
                self.__setattr__(key, np.array([_value for _key, _value in value.items()]))
            if flatten_action_masks:
                action_masks = []
                for am in self.obs['action_mask'].values:
                    action_masks.append(np.concatenate([[1]] + list(am.values())))
                self.action_mask = np.array(action_masks)
            else:
                action_masks = []
                for nation_am in self.obs['action_mask'].values:
                    am = np.array(list(nation_am.values()))
                    action_masks.append(am)
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
                        for nation, nation_dict in self.obs[c].to_dict().items():
                            for key, value in nation_dict.items():
                                new_c = c + '-' + key
                                if new_c not in self.obs.columns:
                                    self.obs[new_c] = np.nan
                                    columns.append(new_c)
                                self.obs.update(pd.DataFrame(data={new_c: [value]}, index=[nation]))

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