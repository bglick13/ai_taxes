import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation
from util.observation_batch import ObservationBatch
from typing import Dict, List, Tuple
from policies.base_neural_net import BaseNeuralNet


class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []  # Discounted rewards
        self.predicted_values = []

    def __getitem__(self, item):
        return (self.states[item].world_map, self.states[item].flat_inputs, self.actions[item], self.old_logprobs[item], self.rewards[item],
                self.predicted_values[item])

    def __len__(self):
        return len(self.states)

    def add_trace(self, states, actions, old_logprobs, rewards, predicted_values):
        self.states += states
        self.actions += actions
        self.old_logprobs += old_logprobs
        self.rewards += rewards
        self.predicted_values += predicted_values

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.old_logprobs[:]
        del self.rewards[:]
        del self.predicted_values[:]


class PPO:
    def __init__(self, env_config, models: Dict[Tuple[str], BaseNeuralNet], lr=0.0003, gamma=0.998, clip_param=0.2, entropy_coef=0.025,
                 value_loss_coef=0.05):
        self.memory = dict()
        self.env_config = env_config
        env = foundation.make_env_instance(**env_config)
        self.models = dict()
        obs = env.reset()
        for key, value in models.items():
            self.models[key] = value()
            self.models[key].build_models(ObservationBatch(obs, key))
            self.memory[key] = Memory()
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.optimizers = dict()
        for key, value in self.models.items():
            self.optimizers[key] = torch.optim.Adam(value.parameters(), lr=self.lr)

    def compute_gae(self, next_value, rewards, masks, values, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = torch.FloatTensor(rewards[step]) + (self.gamma * values[step + 1] * masks[step]).squeeze() - values[step]
            gae = delta + self.gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, key, epochs, batch_size=1, shuffle=True, num_workers=0):
        for _ in range(epochs):
            dataloader = DataLoader(self.memory[key], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for i, batch in enumerate(dataloader):
                world_maps, flat_inputs, actions, old_logprobs, rewards, predicted_values = batch

                world_maps = world_maps.reshape(batch_size * world_maps.shape[1], world_maps.shape[2], world_maps.shape[3], world_maps.shape[4])
                flat_inputs = flat_inputs.reshape(batch_size * flat_inputs.shape[1], flat_inputs.shape[2])
                actions = actions.reshape(-1, 1)
                old_logprobs = old_logprobs.reshape(batch_size * old_logprobs.shape[1], old_logprobs.shape[2])
                predicted_values = predicted_values.reshape(-1, 1)
                rewards = rewards.reshape(-1, 1)

                obs_batch = ObservationBatch([world_maps.float(), flat_inputs.float()])
                self.models[key](obs_batch)
                entropy = self.models[key].dist.entropy().mean()
                new_log_probs = self.models[key].dist.log_prob(actions)
                ratio = (new_log_probs - old_logprobs).exp()
                surr1 = ratio * (rewards - predicted_values)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * (rewards - predicted_values)
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (rewards - predicted_values).pow(2).mean()

                loss = self.value_loss_coef * critic_loss + actor_loss - self.entropy_coef * entropy

                self.optimizers[key].zero_grad()
                loss.backward()
                self.optimizers[key].step()

    def rollout(self, key, n_rollouts, num_steps):
        env = foundation.make_env_instance(**self.env_config)
        for rollout in range(n_rollouts):
            obs = env.reset()
            states, actions, logprobs, rewards, values, done = [], [], [], [], [], []
            for step in range(num_steps):
                obs_batch = ObservationBatch(obs, key)
                self.models[key](obs_batch)
                a = self.models[key].dist.sample()
                value = self.models[key].value.squeeze()
                logprob = self.models[key].dist.log_prob(a)
                action_dict = dict((i, a.detach().cpu().numpy()) for i, a in zip(obs_batch.order, a.argmax(-1)))
                next_obs, rew, is_done, info = env.step(action_dict)

                states.append(obs_batch)
                actions.append(a.argmax(-1))
                logprobs.append(logprob)
                rewards.append(np.array([rew[k] for k in key]))
                values.append(value)
                done.append(is_done['__all__'])

                obs = next_obs
            obs_batch = ObservationBatch(obs, key)
            self.models[key](obs_batch)
            next_value = self.models[key].value
            discounted_rewards = self.compute_gae(next_value, rewards, done, values)
            self.memory[key].add_trace(states, actions, logprobs, discounted_rewards, values)

    def train(self, key_order: List[Tuple[List[str], Dict]], n_training_steps):
        for it in range(n_training_steps):
            for key, spec in key_order:
                self.rollout(key, spec.get('n_rollouts'), spec.get('n_steps_per_rollout'))
                self.update(spec.get('epochs_per_train_step'), spec.get('batch_size'))
                self.memory[key].clear_memory()


