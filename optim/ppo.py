import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation


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
    def __init__(self, env_config, model, lr, betas, gamma, clip_param):
        self.memory = Memory()
        self.env_config = env_config
        self.model = model
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.clip_param = clip_param
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)

    def compute_gae(self, next_value, rewards, masks, values, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def train(self, epochs, batch_size=1, shuffle=True, num_workers=1):
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

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def rollout(self, n_rollouts, num_steps):
        env = foundation.make_env_instance(**self.env_config)
        for rollout in range(n_rollouts):
            obs = env.reset()
            states, actions, logprobs, rewards, values, done = [], [], [], [], [], []
            for step in range(num_steps):
                action_dict = {}
                # TODO: We can probably batchify this in the model definition for speedup
                for agent_idx, agent_obs in obs.items():
                    if agent_idx != 'p':  # We'll do the planner separately
                        self.model(agent_obs)
                        action_dict[agent_idx] = self.model.dist.sample()


