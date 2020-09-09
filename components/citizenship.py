import numpy as np

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
                    'nation_to_idx': self.nations_to_idx}
        else:
            return {}

    def component_step(self):
        world = self.world
        build = []
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
                agent.state['nation'] = self.world.planner.idx_to_nation[action - 1]
                agent.state['nation_idx'] = action - 1
                # TODO: Implement logic to move agent to closest occupy-able location to nation's capitol
                if self.relocate_on_immigrate:
                    pass

            else:
                raise ValueError

        self.builds.append(build)

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "nation": agent.state["nation"],
                "nation_idx": agent.state['nation_idx']
            }

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