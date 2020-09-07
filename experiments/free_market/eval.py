import os, sys

current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..', '..'))

from ai_economist import foundation
from optim.ppo import PPO
from policies.mobile_agent_neural_net import MobileAgentNeuralNet
from experiments.free_market.config import env_config, experiment_name

if __name__ == '__main__':
    env = foundation.make_env_instance(**env_config)
    mobile_agent_model = MobileAgentNeuralNet
    agent_spec = {
        ('0', '1', '2', '3'): mobile_agent_model
    }
    trainer = PPO(env_config, agent_spec)
    trainer.load_weights_from_file()
    train_spec = [
        (('0', '1', '2', '3'),
         {'n_rollouts': 30, 'n_steps_per_rollout': 1000, 'epochs_per_train_step': 16, 'batch_size': 3000})
    ]
    trainer.eval(agent_spec)