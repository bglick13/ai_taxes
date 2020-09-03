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
import torch.multiprocessing as mp

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

    def add_memory(self, memory):
        self.states += memory.states
        self.actions += memory.actions
        self.old_logprobs += memory.old_logprobs
        self.rewards += memory.rewards
        self.advantages += memory.advantages
        self.hcs += memory.hcs

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.old_logprobs[:]
        del self.rewards[:]
        del self.advantages[:]
        del self.hcs[:]


def rollout(env_config, key, constructor, state_dict, n_rollouts, num_steps):
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    t = trange(n_rollouts, desc='Rollout')
    memory = Memory()
    model = constructor(device='cpu')
    model.build_models(ObservationBatch(obs, key))
    model.load_state_dict(state_dict, strict=False)
    for rollout in t:
        obs = env.reset()
        states, actions, logprobs, rewards, values, done, hcs = [], [], [], [], [], [], []
        hc = (torch.zeros(1, len(key), model.lstm_size, device='cpu'),
              torch.zeros(1, len(key), model.lstm_size, device='cpu'))
        for step in range(num_steps):
            obs_batch = ObservationBatch(obs, key)
            hcs.append(hc)
            dist, value, hc = model(obs_batch, hc)
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

        obs_batch = ObservationBatch(obs, key)
        _, next_value, hc = model(obs_batch, hc)
        next_value = next_value.detach().cpu().numpy()
        values = torch.stack(values).detach().cpu().numpy()
        discounted_rewards = compute_gae(next_value, rewards, done, values)
        advantage = (discounted_rewards - values).tolist()
        discounted_rewards = discounted_rewards.tolist()
        memory.add_trace(states, actions, logprobs, discounted_rewards, advantage, hcs)
        return memory


def compute_gae(next_value, rewards, masks, values, tau=0.95, gamma=0.99):
    values = np.vstack((values, next_value.T))
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + (gamma * values[step + 1] * masks[step]).squeeze() - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return np.array(returns)


def join_memories(memories: List[Memory]):
    out = Memory()
    for memory in memories:
        out.add_memory(memory)
    return out


class PPO:
    def __init__(self, env_config, models: Dict[Tuple, BaseNeuralNet], lr=0.0003, gamma=0.998, clip_param=0.2, entropy_coef=0.025,
                 value_loss_coef=0.05, device='cuda'):
        self.memory = dict()
        self.device = device
        self.env_config = env_config
        env = foundation.make_env_instance(**env_config)
        self.models = dict()
        self.model_key_to_constructor = models
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

    def update(self, key, epochs, batch_size=1, shuffle=False, num_workers=0):
        losses = []
        self.models[key].to('cuda')
        for epoch in range(epochs):
            dataloader = DataLoader(self.memory[key], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for i, batch in enumerate(dataloader):
                world_maps, flat_inputs, actions, old_logprobs, rewards, advantages, hcs = batch

                world_maps = world_maps.reshape(batch_size * world_maps.shape[1], world_maps.shape[2], world_maps.shape[3], world_maps.shape[4])
                flat_inputs = flat_inputs.reshape(batch_size * flat_inputs.shape[1], flat_inputs.shape[2])
                actions = actions.reshape(-1, 1).to(self.device)
                old_logprobs = old_logprobs.reshape(batch_size * old_logprobs.shape[1], old_logprobs.shape[2]).to(self.device)
                advantages = torch.stack(advantages).reshape(-1, 1).to(self.device)
                rewards = torch.stack(rewards).reshape(-1, 1).to(self.device)
                hs = hcs[0].permute(1, 0, 2, 3).to(self.device)
                hs = hs.reshape(hs.shape[0], hs.shape[1] * hs.shape[2], -1)
                cs = hcs[1].permute(1, 0, 2, 3).to(self.device)
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
                losses.append(loss.detach().cpu().numpy())
                self.optimizers[key].step()
        return losses

    def train(self, key_order: List[Tuple[Tuple, Dict]], n_training_steps, n_jobs=1):
        t = trange(n_training_steps, desc='Training Iteration', leave=True)
        for it in t:
            for key, spec in key_order:
                constructor = self.model_key_to_constructor[key]
                state_dict = self.models[key].to('cpu').state_dict()
                result = mp.Pool(n_jobs).starmap(rollout, [(self.env_config, key, constructor, state_dict, 1, spec.get('n_steps_per_rollout')) for _ in range(spec.get('n_rollouts'))])
                self.memory[key] = join_memories(result)
                # self.rollout(key, spec.get('n_rollouts'), spec.get('n_steps_per_rollout'))
                losses = self.update(key, spec.get('epochs_per_train_step'), spec.get('batch_size'))
                torch.save(self.models[key].state_dict(), f'{key}_checkpoint_{it}.torch')
                self.memory[key].clear_memory()
                t.set_description(f'Training Iteration (Average Reward: {np.mean(self.memory[key].rewards).round(3)}, Average Loss: {np.mean(losses).round(3)}')
                t.refresh()


