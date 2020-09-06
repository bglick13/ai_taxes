# import torch
from torch import stack, zeros, clamp, min, save, load
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation
from tutorials.utils.plotting import breakdown
from util.observation_batch import ObservationBatch
from typing import Dict, List, Tuple
from policies.base_neural_net import BaseNeuralNet
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from IPython import embed
import torch.multiprocessing as mp
import warnings
import pickle
import os
warnings.filterwarnings("ignore")


class Memory(Dataset):
    def __init__(self):
        self.memories = []
        # self.states = []
        # self.actions = []
        # self.old_logprobs = []
        # self.rewards = []  # Discounted rewards
        # self.advantages = []
        # self.hcs = []

    def __getitem__(self, item):
        return pickle.loads(self.memories[item])

    def __len__(self):
        return len(self.memories)

    def add_trace(self, states, actions, old_logprobs, rewards, advantages, hcs):
        # TODO: Pickle and unpickle to solve some memory issues
        [self.memories.append(pickle.dumps((states[i].world_map, states[i].flat_inputs, actions[i], old_logprobs[i], rewards[i], advantages[i], hcs[i]))) for i in range(len(states))]
        # self.states += states
        # self.actions += actions
        # self.old_logprobs += old_logprobs
        # self.rewards += rewards
        # self.advantages += advantages
        # self.hcs += hcs

    def add_memory(self, memory):
        self.memories += memory.memories
        # self.states += memory.states
        # self.actions += memory.actions
        # self.old_logprobs += memory.old_logprobs
        # self.rewards += memory.rewards
        # self.advantages += memory.advantages
        # self.hcs += memory.hcs

    def clear_memory(self):
        del self.memories[:]
        # del self.states[:]
        # del self.actions[:]
        # del self.old_logprobs[:]
        # del self.rewards[:]
        # del self.advantages[:]
        # del self.hcs[:]


def rollout(env_config, key, constructor, state_dict, n_rollouts, num_steps, _eval=False):
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    t = range(n_rollouts)
    memory = Memory()
    model = constructor(device='cpu')
    model.build_models(ObservationBatch(obs, key))
    model.load_state_dict(state_dict, strict=False)
    for rollout in t:
        obs = env.reset(force_dense_logging=_eval)
        states, actions, logprobs, rewards, values, done, hcs = [], [], [], [], [], [], []
        hc = (zeros(1, len(key), model.lstm_size, device='cpu'),
              zeros(1, len(key), model.lstm_size, device='cpu'))
        for step in range(num_steps):
            obs_batch = ObservationBatch(obs, key)
            hcs.append(hc)
            # if _eval:
            #     dist, value, hc = model(obs_batch, hc, det=True)
            #     a = np.array(dist).reshape(-1,1)
            #     action_dict = {i: a[i] for i in range(len(a))}
            #     actions.append(a)
            #     logprob = 0
            # else:
            dist, value, hc = model(obs_batch, hc, det=False)
            a = dist.sample().detach()
            action_dict = dict((i, a.detach().cpu().numpy()) for i, a in zip(obs_batch.order, a.argmax(-1)))
            actions.append(a.argmax(-1).detach())
            # print(actions)
            logprob = dist.log_prob(a).detach()
            logprobs.append(logprob)
            hc = (hc[0].detach(), hc[1].detach())
            value = value.squeeze()
            next_obs, rew, is_done, info = env.step(action_dict)
            states.append(obs_batch)
            rewards.append(np.array([rew[k] for k in key]))
            values.append(value)
            done.append(is_done['__all__'])

            obs = next_obs

        obs_batch = ObservationBatch(obs, key)
        _, next_value, hc = model(obs_batch, hc)
        next_value = next_value.detach().cpu().numpy()
        values = stack(values).detach().cpu().numpy()
        discounted_rewards = compute_gae(next_value, rewards, done, values)
        advantage = (discounted_rewards - values).tolist()
        discounted_rewards = discounted_rewards.tolist()
        memory.add_trace(states, actions, logprobs, discounted_rewards, advantage, hcs)
        if _eval:
            return memory, env.previous_episode_dense_log
        else:
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
            self.optimizers[key] = Adam(value.parameters(), lr=self.lr)

    def load_weights_from_file(self, experiment_name):
        for key, value in self.models.items():
            weights = load(os.path.join(f'experiments/{experiment_name}/weights/{key}_final.torch'))
            self.models[key].load_state_dict(weights)

    def update(self, key, epochs, batch_size=1, shuffle=False, num_workers=0):
        losses = []
        all_rewards = []
        self.models[key].to('cuda')
        for epoch in range(epochs):
            dataloader = DataLoader(self.memory[key], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            for i, batch in enumerate(dataloader):
                world_maps, flat_inputs, actions, old_logprobs, rewards, advantages, hcs = batch
                all_rewards += rewards
                world_maps = world_maps.reshape(batch_size * world_maps.shape[1], world_maps.shape[2], world_maps.shape[3], world_maps.shape[4])
                flat_inputs = flat_inputs.reshape(batch_size * flat_inputs.shape[1], flat_inputs.shape[2])
                actions = actions.reshape(-1, 1).to(self.device)
                old_logprobs = old_logprobs.reshape(batch_size * old_logprobs.shape[1], old_logprobs.shape[2]).to(self.device)
                advantages = stack(advantages).reshape(-1, 1).to(self.device)
                rewards = stack(rewards).reshape(-1, 1).to(self.device)
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
                surr2 = clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = - min(surr1, surr2).mean()
                critic_loss = (rewards - value).pow(2).mean()

                loss = self.value_loss_coef * critic_loss + actor_loss - self.entropy_coef * entropy
                # print(f'Epoch: {epoch} Loss: {loss}')
                self.optimizers[key].zero_grad()
                loss.backward()
                losses.append(loss.detach().cpu().numpy())
                self.optimizers[key].step()
        return losses, all_rewards

    def train(self, key_order: List[Tuple[Tuple, Dict]], n_training_steps, experiment_name, n_jobs=1):
        writer = SummaryWriter()
        t = trange(n_training_steps, desc='Training Iteration', leave=True)
        if not os.path.isdir(f'weights/temp'):
            os.makedirs(f'weights/temp')
        for it in t:
            for key, spec in key_order:
                constructor = self.model_key_to_constructor[key]
                state_dict = self.models[key].to('cpu').state_dict()
                with mp.Pool(n_jobs) as pool:
                    result = pool.starmap(rollout, [(self.env_config, key, constructor, state_dict, 1, spec.get('n_steps_per_rollout')) for _ in range(spec.get('n_rollouts'))])
                self.memory[key] = join_memories(result)
                # self.rollout(key, spec.get('n_rollouts'), spec.get('n_steps_per_rollout'))
                losses, all_rewards = self.update(key, spec.get('epochs_per_train_step'), spec.get('batch_size'))
                writer.add_scalar('Loss/train', np.mean(losses), it)
                writer.add_scalar('Reward/train', stack(all_rewards).mean().detach().cpu().numpy().round(3), it)
                save(self.models[key].state_dict(), f'weights/temp/{key}_checkpoint_{it}.torch.temp')
                t.set_description(f'Training Iteration (Average Reward: {stack(all_rewards).mean().detach().cpu().numpy().round(3)}, Average Loss: {np.mean(losses).round(3)}')
                self.memory[key].clear_memory()
                t.refresh()
        for key, spec in key_order:
            save(self.models[key].state_dict(), f'weights/{key}_final.torch')

    def eval(self, key_order: List[Tuple[Tuple, Dict]]):
        # TODO: Connor implement this. Can probably use rollout function above
        dense_logs = list()
        for key, spec in key_order.items():
            constructor = self.model_key_to_constructor[key]
            state_dict = self.models[key].to('cpu').state_dict()
            env, log = rollout(self.env_config, key, constructor, state_dict, n_rollouts=1, num_steps=1000, _eval=True)
            dense_logs.append(log)
            b = breakdown(log)
        embed()