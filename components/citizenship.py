import numpy as np
from IPython import embed

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class OpenBorderCitizenship(BaseComponent):

    name = "OpenBorderCitizenship"
    component_type = "Citizenship"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(self, *args, nations=None, relocate_on_immigrate=False, **kwargs):
        super().__init__(*args, **kwargs)
        if nations is None:
            self.nations = ['foo_land', 'bar_land']
        else:
            self.nations = nations
        self.n_nations = len(self.nations)
        self.nations_to_idx = dict((n, i) for i, n in enumerate(self.nations))
        self.idx_to_nations = dict((i, n) for i, n in enumerate(self.nations))
        self.relocate_on_immigrate = relocate_on_immigrate
        self.citizenship_count = dict((n, 0) for n in self.nations)
        # embed()

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == 'BasicMobileAgent':
            return self.n_nations
        else:
            return None

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == 'BasicMobileAgent':
            return {'nation': -1}
        elif agent_cls_name == 'BasicPlanner':
            return {'nations': self.nations,
                    'nation_to_idx': self.nations_to_idx,
                    'idx_to_nation': self.idx_to_nations,
                    'citizenship_count': self.citizenship_count}
        else:
            return {}

    def component_step(self):
        world = self.world
        self.citizenship_count = dict((n, 0) for n in self.nations)
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Immigrate!
            elif 1 <= action <= self.n_nations + 1:
                agent.state['nation'] = self.world.planner.state['idx_to_nation'][action - 1]
                agent.state['nation_idx'] = action - 1
                
                nation = agent.state['nation']
                nation_capital_loc = self.world.capital_locations[nation]
                r = nation_capital_loc[0] + np.random.randint(0, 2)
                c = nation_capital_loc[1] + np.random.randint(0, 2)
                n_tries = 0

                # TODO: Make sure that an agent cannot spawn in a differen't nation's zone(s).
                #       This could happen if width != height. 
                while not self.world.can_agent_occupy(r, c, agent):
                    r = nation_capital_loc[0] + np.random.randint(0, 2)
                    c = nation_capital_loc[1] + np.random.randint(0, 2)
                    n_tries += 1
                    if n_tries > 200:
                        raise TimeoutError
                self.world.set_agent_loc(agent, r, c)

                if self.relocate_on_immigrate:
                    pass

            else:
                raise ValueError
        for agent in world.agents:
            self.citizenship_count[agent.state['nation']] += 1
        self.world.planner.state['citizenship_count'] = self.citizenship_count

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "nation": agent.state["nation"],
                "nation_idx": agent.state['nation_idx']
            }
        obs_dict['p'] = dict(citizenship_count=self.citizenship_count)

        return obs_dict

    def generate_masks(self, completions=0):
        masks = {}
        is_first_day = self.world.planner.state['tax_cycle_pos'] == 1
        is_first_period = len(self.world.planner.state['taxes']) < self.world.planner.state['period']
        for agent in self.world.agents:
            masks[agent.idx] = np.ones(self.n_nations) if (is_first_day and not is_first_period) else np.zeros(self.n_nations)
        return masks

    def get_metrics(self):
        pass

    def get_dense_log(self):
        pass

    def additional_reset_steps(self):
        for agent in self.world.agents:
            n = np.random.choice(self.nations)
            agent.state['nation'] = n
            agent.state['nation_idx'] = self.nations_to_idx[n]