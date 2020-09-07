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
    train_spec = [
        (('0', '1', '2', '3'),
         {'n_rollouts': 30, 'n_steps_per_rollout': 200, 'epochs_per_train_step': 16, 'batch_size': 1000})
    ]
    trainer.train(train_spec, 100, experiment_name, n_jobs=4)