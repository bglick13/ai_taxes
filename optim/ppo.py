import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation
from util.observation_batch import ObservationBatch

class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []  # Discounted rewards
        self.predicted_values = []
        self.done = []

    def __getitem__(self, item):
        return (self.states[item], self.actions[item], self.old_logprobs[item], self.rewards[item],
                self.predicted_values[item], self.done[item])

    def __len__(self):
        return len(self.states)

    def add_trace(self, states, actions, old_logprobs, rewards, predicted_values, done):
        self.states += states
        self.actions += actions
        self.old_logprobs += old_logprobs
        self.rewards += rewards
        self.predicted_values += predicted_values
        self.done += done

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.old_logprobs[:]
        del self.rewards[:]
        del self.predicted_values[:]
        del self.done[:]


class PPO:
    def __init__(self, env_config, model, lr=0.0003, gamma=0.998, clip_param=0.2, entropy_coef=0.025,
                 value_loss_coef=0.05):
        self.memory = Memory()
        self.env_config = env_config
        env = foundation.make_env_instance(**env_config)
        self.model = model()
        self.model.build_models(ObservationBatch(env.reset(), ['0', '1', '2', '3']))
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_gae(self, next_value, rewards, masks, values, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, epochs, batch_size=1, shuffle=True, num_workers=1):
        for _ in range(epochs):
            dataloader = DataLoader(self.memory, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for i, batch in enumerate(dataloader):
                states, actions, old_logprobs, rewards, predicted_values, done = batch
                self.model(states)
                entropy = self.model.dist().entropy().mean()
                new_log_probs = self.model.dist().log_prob(actions)
                ratio = (new_log_probs - old_logprobs).exp()
                surr1 = ratio * (rewards - predicted_values)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * (rewards - predicted_values)
                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (rewards - predicted_values).pow(2).mean()

                loss = self.value_loss_coef * critic_loss + actor_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def rollout(self, n_rollouts, num_steps):
        env = foundation.make_env_instance(**self.env_config)
        for rollout in range(n_rollouts):
            obs = env.reset()
            states, actions, logprobs, rewards, values, done = [], [], [], [], [], []
            for step in range(num_steps):
                obs_batch = ObservationBatch(obs, ['0', '1', '2', '3'])
                self.model(obs_batch)
                a = self.model.dist.sample()
                value = self.model.value
                logprob = self.model.dist.log_prob(a)
                action_dict = dict((i, a.detach().cpu().numpy()) for i, a in zip(obs_batch.order, a.argmax(-1)))
                next_obs, rew, is_done, info = env.step(action_dict)

                states.append(obs_batch)
                actions.append(a.argmax(-1))
                logprobs.append(logprob)
                rewards.append(rew)
                values.append(value)
                done.append(is_done)

                obs = next_obs



