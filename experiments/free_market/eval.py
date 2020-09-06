from ai_economist import foundation
from optim.ppo import PPO
from policies.mobile_agent_neural_net import MobileAgentNeuralNet
from experiments.free_market.config import env_config

if __name__ == '__main__':
    env = foundation.make_env_instance(**env_config)
    mobile_agent_model = MobileAgentNeuralNet