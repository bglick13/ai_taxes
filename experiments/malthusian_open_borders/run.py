import os, sys
current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..', '..'))

from optim.ppo import PPO
from policies.mobile_agent_neural_net import MobileAgentNeuralNet
from policies.tax_planner_neural_net import TaxPlannerNeuralNet
from experiments.malthusian_open_borders.config import env_config, experiment_name

if __name__ == '__main__':
    mobile_agent_model = MobileAgentNeuralNet
    tax_planner_agent_model = TaxPlannerNeuralNet
    agent_spec = {
        ('0', '1', '2', '3', '4', '5'): dict(constructor=mobile_agent_model, lr=0.0003, entropy_loss_coef=0.025),
        ('p', ): dict(constructor=tax_planner_agent_model, lr=0.0001, entropy_loss_coef=0.1)
    }
    env_config['dense_log_frequency'] = None
    trainer = PPO(env_config, agent_spec)
    train_spec = [
        (('0', '1', '2', '3', '4', '5'),
         {'n_rollouts': 60, 'n_steps_per_rollout': 1000, 'epochs_per_train_step': 16, 'batch_size': 1000, 'rollouts_per_job': 1}),
        (('p', ),
         {'n_rollouts': 60, 'n_steps_per_rollout': 1000, 'epochs_per_train_step': 16, 'batch_size': 1000, 'rollouts_per_job': 1})
    ]
    n_epochs = 200
    # trainer.load_weights_from_file(from_checkpoint=True)
    print(f'Training for {train_spec[0][1]["n_rollouts"] * train_spec[0][1]["n_steps_per_rollout"] * len(train_spec[0][0]) * n_epochs} total steps')
    trainer.train(train_spec, n_epochs, n_jobs=6, starting_it=0)