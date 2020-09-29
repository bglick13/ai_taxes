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

    def __init__(self, *args, nations=None, nations_to_idx=None, idx_to_nations=None, relocate_on_immigrate=True,
                 labor_cost=10.0, annealing_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        if nations is None:
            self.nations = ['foo_land', 'bar_land']
        else:
            self.nations = nations
        self.n_nations = len(self.nations)
        self.nations_to_idx = nations_to_idx
        self.idx_to_nations = idx_to_nations
        self.relocate_on_immigrate = relocate_on_immigrate
        self.labor_cost = labor_cost
        self.citizenship_count = dict((n, 0) for n in self.nations)
        self.immigrations = []
        self.annealing_steps = annealing_steps
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
        immigrations = []
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
                immigrations.append([dict(
                    agent_idx=agent.idx,
                    from_nation=agent.state['nation'],
                    to_nation=self.idx_to_nations[action - 1]
                )])
                agent.state["endogenous"]["Labor"] += self.labor_cost
                agent.state['nation'] = self.idx_to_nations[action - 1]
                agent.state['nation_idx'] = action - 1
                
                agent_nation_zone = self.world.nation_zones[agent.state['nation']]
                r = np.random.randint(agent_nation_zone[0][1], agent_nation_zone[2][1] + 1)
                c = np.random.randint(agent_nation_zone[0][0], agent_nation_zone[1][0] + 1)
                n_tries = 0

                # TODO: Make sure that an agent cannot spawn in a different nation's zone(s).
                #       This could happen if width != height.
                if self.relocate_on_immigrate:
                    tmp = []
                    while not self.world.can_agent_occupy(r, c, agent):
                        r = np.random.randint(agent_nation_zone[0][1], agent_nation_zone[2][1] + 1)
                        c = np.random.randint(agent_nation_zone[0][0], agent_nation_zone[1][0] + 1)
                        n_tries += 1
                        tmp.append((r, c))
                        if n_tries > 200:
                            print(tmp)
                            raise TimeoutError
                    self.world.set_agent_loc(agent, r, c)

            else:
                raise ValueError
        self.immigrations.append(immigrations)
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
        self.step += 1
        masks = {}
        if self.annealing_steps is None:
            cutoff = 0.01
        else:
            cutoff = 0.5 - min(((completions / self.annealing_steps) * 0.49), 0.49)
        is_first_half = (self.world.planner.state['tax_cycle_pos'] / self.world.planner.state['period']) <= cutoff
        # is_first_day = self.world.planner.state['tax_cycle_pos'] == 1
        # is_first_period = len(self.world.planner.state['taxes']) < self.world.planner.state['period']

        for agent in self.world.agents:
            masks[agent.idx] = np.ones(self.n_nations) if is_first_half else np.zeros(self.n_nations)
            masks[agent.idx][self.nations_to_idx[agent.state['nation']]] = 0  # Agent cannot immigrate to current nation
        return masks

    def get_metrics(self):
        pass

    def get_dense_log(self):
        return self.immigrations

    def additional_reset_steps(self):
        self.step = 0
        self.immigrations = []
        for agent in self.world.agents:
            n = np.random.choice(self.nations)
            agent.state['nation'] = n
            agent.state['nation_idx'] = self.nations_to_idx[n]

        for agent in self.world.agents:
            agent_nation_zone = self.world.nation_zones[agent.state['nation']]
            r = np.random.randint(agent_nation_zone[0][1], agent_nation_zone[2][1] + 1)
            c = np.random.randint(agent_nation_zone[0][0], agent_nation_zone[1][0] + 1)
            n_tries = 0

            # TODO: Make sure that an agent cannot spawn in a different nation's zone(s).
            #       This could happen if width != height.
            tmp = []
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(agent_nation_zone[0][1], agent_nation_zone[2][1] + 1)
                c = np.random.randint(agent_nation_zone[0][0], agent_nation_zone[1][0] + 1)
                n_tries += 1
                tmp.append((r, c))
                if n_tries > 200:
                    print(tmp)
                    raise TimeoutError
            self.world.set_agent_loc(agent, r, c)