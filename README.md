# TODO:
1) Extend marginal tax component to have the redistribution scheme be a learnable part of the policy
2) Use Malthusian RL to represent several 'nations' (instantiated as individual tax planners)
    - This will require extending the env code to allow for multiple planners
    - Each nation will have its own social equity goals (which will include the size of its economy)
    - Each agent will have the option of taking an action to immigrate to a different nation
        - We can extend this even further by comparing open border policies to a bid/ask system similar to resource trading
    - We can also extend this even further to simulate international trade policy