import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation
from util.observation_batch import ObservationBatch
from typing import Dict, List, Tuple
from policies.base_neural_net import BaseNeuralNet
from tqdm import tqdm, trange
import copy

torch.autograd.set_detect_anomaly(True)


class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []  # Discounted rewards
        self.advantages = []
        self.hcs = []

    def __getitem__(self, item):
        return (self.states[item].world_map, self.states[item].flat_inputs, self.actions[item], self.old_logprobs[item], self.rewards[item],
                self.advantages[item], self.hcs[item])

    def __len__(self):
        return len(self.states)

    def add_trace(self, states, actions, old_logprobs, rewards, advantages, hcs):
        self.states += states
        self.actions += actions
        self.old_logprobs += old_logprobs
        self.rewards += rewards
        self.advantages += advantages
        self.hcs += hcs

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.old_logprobs[:]
        del self.rewards[:]
        del self.advantages[:]
        del self.hcs[:]


class PPO:
    def __init__(self, env_config, models: Dict[Tuple, BaseNeuralNet], lr=0.0003, gamma=0.998, clip_param=0.2, entropy_coef=0.025,
                 value_loss_coef=0.05, device='cuda'):
        self.memory = dict()
        self.device = device
        self.env_config = env_config
        env = foundation.make_env_instance(**env_config)
        self.models = dict()
        obs = env.reset()
        for key, value in models.items():
            self.models[key] = value(device=self.device)
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
        values = np.vstack((values, next_value.T))
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + (self.gamma * values[step + 1] * masks[step]).squeeze() - values[step]
            gae = delta + self.gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return np.array(returns)

    def update(self, key, epochs, batch_size=1, shuffle=True, num_workers=0):
        for epoch in range(epochs):
            dataloader = DataLoader(self.memory[key], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for i, batch in enumerate(dataloader):
                world_maps, flat_inputs, actions, old_logprobs, rewards, advantages, hcs = batch

                world_maps = world_maps.reshape(batch_size * world_maps.shape[1], world_maps.shape[2], world_maps.shape[3], world_maps.shape[4])
                flat_inputs = flat_inputs.reshape(batch_size * flat_inputs.shape[1], flat_inputs.shape[2])
                actions = actions.reshape(-1, 1)
                old_logprobs = old_logprobs.reshape(batch_size * old_logprobs.shape[1], old_logprobs.shape[2])
                advantages = torch.stack(advantages).reshape(-1, 1).to(self.device)
                rewards = torch.stack(rewards).reshape(-1, 1).to(self.device)
                hs = torch.stack([hc[0] for hc in hcs]).permute(1, 0, 2, 3)
                hs = hs.reshape(hs.shape[0], hs.shape[1] * hs.shape[2], -1)
                cs = torch.stack([hc[1] for hc in hcs]).permute(1, 0, 2, 3)
                cs = cs.reshape(cs.shape[0], cs.shape[1] * cs.shape[2], -1)
                hcs = (hs, cs)

                obs_batch = ObservationBatch([world_maps.float(), flat_inputs.float()])
                dist, value, hc = self.models[key](obs_batch, hcs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions)
                ratio = (new_log_probs - old_logprobs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (rewards - value).pow(2).mean()

                loss = self.value_loss_coef * critic_loss + actor_loss - self.entropy_coef * entropy
                print(f'Epoch: {epoch} Loss: {loss}')
                self.optimizers[key].zero_grad()
                loss.backward()
                self.optimizers[key].step()

    def rollout(self, key, n_rollouts, num_steps):
        env = foundation.make_env_instance(**self.env_config)
        rewards = []
        t = trange(n_rollouts, desc='Rollout', leave=True)
        for rollout in t:
            obs = env.reset()
            states, actions, logprobs, rewards, values, done, hcs = [], [], [], [], [], [], []
            hc = (torch.zeros(1, len(key), self.models[key].lstm_size, device=self.device),
                  torch.zeros(1, len(key), self.models[key].lstm_size, device=self.device))
            for step in range(num_steps):
                obs_batch = ObservationBatch(obs, key)
                hcs.append(hc)
                dist, value, hc = self.models[key](obs_batch, hc)
                hc = (hc[0].detach(), hc[1].detach())
                a = dist.sample().detach()
                value = value.squeeze()
                logprob = dist.log_prob(a).detach()
                action_dict = dict((i, a.detach().cpu().numpy()) for i, a in zip(obs_batch.order, a.argmax(-1)))
                next_obs, rew, is_done, info = env.step(action_dict)
                states.append(obs_batch)
                actions.append(a.argmax(-1).detach())
                logprobs.append(logprob)
                rewards.append(np.array([rew[k] for k in key]))
                values.append(value)
                done.append(is_done['__all__'])

                obs = next_obs
            t.set_description(f'Rollout (Average Marginal Reward: {np.mean(rewards).round(3)})')
            t.refresh()
            obs_batch = ObservationBatch(obs, key)
            _, next_value, hc = self.models[key](obs_batch, hc)
            next_value = next_value.detach().cpu().numpy()
            values = torch.stack(values).detach().cpu().numpy()
            discounted_rewards = self.compute_gae(next_value, rewards, done, values)
            advantage = (discounted_rewards - values).tolist()
            discounted_rewards = discounted_rewards.tolist()
            self.memory[key].add_trace(states, actions, logprobs, discounted_rewards, advantage, hcs)

    def train(self, key_order: List[Tuple[Tuple, Dict]], n_training_steps):
        for it in range(n_training_steps):
            print(f'Training step {it}/{n_training_steps}')
            for key, spec in key_order:
                print(f'Training {key}')
                self.rollout(key, spec.get('n_rollouts'), spec.get('n_steps_per_rollout'))
                self.update(key, spec.get('epochs_per_train_step'), spec.get('batch_size'))
                self.memory[key].clear_memory()


