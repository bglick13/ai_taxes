import os, sys, argparse

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..'))

from ai_economist import foundation
from optim.ppo import PPO
from policies.mobile_agent_neural_net import MobileAgentNeuralNet
    
def main(weight_file):
    env_config = {
        # ===== SCENARIO CLASS =====
        # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
        # The environment object will be an instance of the Scenario class.
        'scenario_name': 'layout_from_file/simple_wood_and_stone',

        # ===== COMPONENTS =====
        # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
        #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
        #   {component_kwargs} is a dictionary of kwargs passed to the Component class
        # The order in which components reset, step, and generate obs follows their listed order below.
        'components': [
            # (1) Building houses
            ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
            # (2) Trading collectible resources
            ('ContinuousDoubleAuction', {'max_num_orders': 5}),
            # (3) Movement and resource collection
            ('Gather', {}),
            # ('PeriodicBracketTax', {})
        ],

        # ===== SCENARIO CLASS ARGUMENTS =====
        # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
        'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
        'starting_agent_coin': 10,
        'fixed_four_skill_and_loc': True,

        # ===== STANDARD ARGUMENTS ======
        # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
        'n_agents': 4,  # Number of non-planner agents (must be > 1)
        'world_size': [25, 25],  # [Height, Width] of the env world
        'episode_length': 1000,  # Number of timesteps per episode

        # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
        # Otherwise, the policy selects only 1 action.
        'multi_action_mode_agents': False,
        'multi_action_mode_planner': True,

        # When flattening observations, concatenate scalar & vector observations before output.
        # Otherwise, return observations with minimal processing.
        'flatten_observations': False,
        # When Flattening masks, concatenate each action subspace mask into a single array.
        # Note: flatten_masks = True is required for masking action logits in the code below.
        'flatten_masks': True,
    }

    env = foundation.make_env_instance(**env_config)
    mobile_agent_model = MobileAgentNeuralNet
    agent_spec = {
        ('0', '1', '2', '3'): mobile_agent_model
    }
    trainer = PPO(env_config, agent_spec)
    if weight_file == None:
        train_spec = [
            (('0', '1', '2', '3'),
             {'n_rollouts': 30, 'n_steps_per_rollout': 200, 'epochs_per_train_step': 16, 'batch_size': 3000})
        ]
        trainer.train(train_spec, 100, n_jobs=4)
    else:
        trainer.load_weights_from_file(weight_file)

    trainer.eval(agent_spec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_file', type=str, default = None)
    args = parser.parse_args()
    main(args.weight_file)