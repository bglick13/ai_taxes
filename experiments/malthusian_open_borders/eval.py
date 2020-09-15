import os, sys

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..', '..'))

from ai_economist import foundation
from optim.ppo import PPO
from policies.mobile_agent_neural_net import MobileAgentNeuralNet
from policies.tax_planner_neural_net import TaxPlannerNeuralNet

from experiments.malthusian_open_borders.config import env_config, experiment_name

if __name__ == '__main__':
    mobile_agent_model = MobileAgentNeuralNet
    tax_planner_agent_model = TaxPlannerNeuralNet

    agent_spec = {
        ('0', '1', '2', '3'): mobile_agent_model,
        ('p',): tax_planner_agent_model
    }
    trainer = PPO(env_config, agent_spec)
    trainer.load_weights_from_file()
    trainer.eval(agent_spec)