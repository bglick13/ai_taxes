experiment_name = 'malthusian_open_borders'
nations = ['foo_land', 'bar_land']
nations_to_idx = dict((n, i) for i, n in enumerate(nations))
idx_to_nations = dict((i, n) for i, n in enumerate(nations))
env_config = {
        # ===== SCENARIO CLASS =====
        # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
        # The environment object will be an instance of the Scenario class.
        'scenario_name': "malthusian_quadrant/simple_wood_and_stone",

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
            ("OpenBorderCitizenship", {'nations': nations, 'nations_to_idx': nations_to_idx, 'idx_to_nations': idx_to_nations, 'annealing_steps': 9000}),
            ('MalthusianPeriodicBracketTax', {'nations': nations, 'nations_to_idx': nations_to_idx, 'idx_to_nations': idx_to_nations,
                                              'tax_annealing_schedule': [9000, 0.01]
                                              })
        ],
        'nations': nations,
        'nations_to_idx': nations_to_idx,
        'idx_to_nations': idx_to_nations,
        # ===== SCENARIO CLASS ARGUMENTS =====
        # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
        'starting_agent_coin': 20,

        # ===== STANDARD ARGUMENTS ======
        # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
        'n_agents': 6,  # Number of non-planner agents (must be > 1)
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
        'flatten_masks': False,
    }