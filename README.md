# TODO:
1) Extend marginal tax component to have the redistribution scheme be a learnable part of the policy
2) Use Malthusian RL to represent several 'nations' (instantiated as individual tax planners)
    - This will require extending the env code to allow for multiple planners
        - We need a new endogenous resource "citizenship"
        - Add a landmark entity describing the location of a nation's capitol - used to relocate immigrating agents
        - A new component "migrate" will add the citizenship state and allow agents to change their citizenship
        - We probably need a new agent component as well so the observations can be nation-centric (like how current taxes work for mobile agents)
        - A new component "multi_nation_bracketed_tax" will apply tax schemes to only the citizens of the given nation
    - Each nation will have its own social equity goals (which will include the size of its economy)
    - Each agent will have the option of taking an action to immigrate to a different nation
        - We can extend this even further by comparing open border policies to a bid/ask system similar to resource trading
    - We can also extend this even further to simulate international trade policy