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
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import warnings
import pickle
import os, sys
from scenarios import *
from components import *
from entities import *


current_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_path, '..'))
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


def rollout(env_config, keys, train_key, constructors, state_dicts, n_rollouts, num_steps, _eval=False):
    """

    :param env_config:
    :param keys: All the keys used in PPO (agents + planner)
    :param train_key: The key we want to store memories for
    :param constructors: dict mapping [key, model_constructor_function)
    :param state_dicts: dict mapping [key, state_dict]
    :param n_rollouts:
    :param num_steps:
    :param _eval:
    :return:
    """
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    t = range(n_rollouts)
    memory = Memory()
    models = dict((key, constructors[key](device='cpu')) for key in keys)
    for key, model in models.items():
        model.build_models(ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True))
        model.load_state_dict(state_dicts[key], strict=False)
    for rollout in t:
        obs = env.reset(force_dense_logging=_eval)
        states, actions, logprobs, rewards, values, done, hcs = [], [], [], [], [], [], []
        model_hcs = dict((key, (zeros(1, len(key), models[key].lstm_size, device='cpu'),
                          zeros(1, len(key), models[key].lstm_size, device='cpu'))) for key in keys)
        for step in range(num_steps):
            obs_batches = dict((key, ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True)) for key in keys)
            hcs.append(model_hcs[train_key])
            action_dict = dict()
            for key in keys:
                dist, value, hc = models[key](obs_batches[key], model_hcs[key], det=False)
                hc = (hc[0].detach(), hc[1].detach())
                model_hcs[key] = hc
                a = dist.sample().detach()
                if key == train_key:
                    actions.append(a.detach())
                    logprob = dist.log_prob(a).detach()
                    logprobs.append(logprob)
                    value = value.squeeze()
                    values.append(value)
                if 'p' in key:
                    n_nations = len(obs_batches[key].order)
                    n_actions = a.shape[1] // n_nations
                    p_array = np.concatenate([_a.detach().cpu().numpy()[i * n_actions: (i + 1) * n_actions] for i, _a in enumerate(a)])
                    action_dict.update(dict(p=p_array))
                else:
                    action_dict.update(dict((i, _a.detach().cpu().numpy()) for i, _a in zip(obs_batches[key].order, a)))

            next_obs, rew, is_done, info = env.step(action_dict)
            states.append(obs_batches[train_key])
            rewards.append(np.array([rew[k] for k in train_key]))

            done.append(is_done['__all__'])

            obs = next_obs

        obs_batch = ObservationBatch(obs, train_key, flatten_action_masks=False if 'p' in train_key else True)
        _, next_value, hc = models[train_key](obs_batch, model_hcs[train_key])
        next_value = next_value.detach().cpu().numpy()
        values = stack(values).detach().cpu().numpy()
        discounted_rewards = compute_gae(next_value, rewards, done, values)
        advantage = (discounted_rewards - values).tolist()
        discounted_rewards = discounted_rewards.tolist()
        memory.add_trace(states, actions, logprobs, discounted_rewards, advantage, hcs)
    if _eval:
        return memory, env.previous_episode_dense_log, [agent.state['build_skill'] for agent in env.world.agents]
    else:
        return memory


def compute_gae(next_value, rewards, masks, values, tau=0.98, gamma=0.998):
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
            self.models[key].build_models(ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True))
            self.memory[key] = Memory()
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.optimizers = dict()
        for key, value in self.models.items():
            self.optimizers[key] = Adam(value.parameters(), lr=self.lr)

    def load_weights_from_file(self):
        for key, value in self.models.items():
            weights = load(os.path.join(current_file_path, '..', 'experiments', 'free_market', 'weights', str(key)+'_final.torch'))    
            #weights = load(os.path.join(f'weights/{key}_final.torch'))
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

                # Reshape: Double check this is right
                world_maps = world_maps.reshape(batch_size * world_maps.shape[1], world_maps.shape[2], world_maps.shape[3], world_maps.shape[4])
                flat_inputs = flat_inputs.reshape(batch_size * flat_inputs.shape[1], flat_inputs.shape[2])
                actions = actions.reshape(-1, 1).to(self.device)
                old_logprobs = old_logprobs.reshape(batch_size * old_logprobs.shape[1], 1).to(self.device)
                advantages = stack(advantages).reshape(-1, 1).to(self.device)
                rewards = stack(rewards).reshape(-1, 1).to(self.device)
                hs = hcs[0].squeeze()
                cs = hcs[1].squeeze()
                hs = hs.reshape(hs.shape[0] * hs.shape[1], hs.shape[2]).to(self.device).unsqueeze(0)  # (batch_size, n_agents, hidden_size)
                cs = cs.reshape(cs.shape[0] * cs.shape[1], cs.shape[2]).to(self.device).unsqueeze(0)  # (batch_size, n_agents, hidden_size)

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
                # clip_grad_norm_(self.models[key].parameters(), 10)
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
                # constructor = self.model_key_to_constructor[key]
                state_dicts = dict((k, v.to('cpu').state_dict()) for k, v in self.models.items())
                rollouts_per_jobs = spec.get('rollouts_per_job', 1)

                # rollout(self.env_config, np.array(key_order)[:, 0], key, self.model_key_to_constructor, state_dicts,
                #         n_rollouts=1, num_steps=1000)

                with mp.Pool(n_jobs) as pool:
                    result = pool.starmap(rollout, [(self.env_config, np.array(key_order)[:, 0], key, self.model_key_to_constructor, state_dicts, rollouts_per_jobs,
                                                     spec.get('n_steps_per_rollout')) for _ in range(spec.get('n_rollouts'))])
                self.memory[key] = join_memories(result)

                losses, all_rewards = self.update(key, spec.get('epochs_per_train_step'), spec.get('batch_size'))
                writer.add_scalar('Loss/train', np.mean(losses), it)
                writer.add_scalar('Reward/train', stack(all_rewards).mean().detach().cpu().numpy().round(3), it)
                save(self.models[key].state_dict(), f'weights/temp/{key}_checkpoint_{it}.torch.temp')
                t.set_description(f'Training Iteration (Average Reward: {stack(all_rewards).mean().detach().cpu().numpy().round(3)}, Average Loss: {np.mean(losses).round(3)}')
                self.memory[key].clear_memory()
                t.refresh()
            if (it+1) % 10 == 0:
                self.eval(key_order, mode='a')
        for key, spec in key_order:
            save(self.models[key].state_dict(), f'weights/{key}_final.torch')

    def eval(self, key_order: List[Tuple[Tuple, Dict]], mode='w+'):
        dense_logs = list()
        for key, spec in key_order.items():
            constructor = self.model_key_to_constructor[key]
            state_dict = self.models[key].to('cpu').state_dict()
            _, log, skills = rollout(self.env_config, np.array(key_order)[:, 0], key, constructor, state_dict, n_rollouts=1, num_steps=1000, _eval=True)
            dense_logs.append(log)
            b = breakdown(log)

        incomes = list(np.round(b[1]['Total'], 3))
        percent_income_from_build = list(np.round(b[1]['Build']/b[1]['Total'], 3))
        correlation = np.corrcoef(percent_income_from_build, skills)[0, 1]
        endowments = list(b[2])
        productivity = sum(incomes)
        equality = sum([log['rewards'][i]['p'] for i in range(len(log['rewards']))]) / productivity
        # equality = foundation.scenarios.utils.get_equality(endowments)
        
        rewards = [[log['rewards'][i][agent] for i in range(len(log['rewards']))] for agent in log['rewards'][0].keys()]
        total_reward_per_timestep = [[rewards[0][0]],[rewards[1][0]],[rewards[2][0]],[rewards[3][0]]]
        for i in range(1, len(rewards[0])):
            for j in range(len(rewards)-1):
                total_reward_per_timestep[j].append(rewards[j][i] + total_reward_per_timestep[j][i-1])
        total_reward_per_timestep = np.array(total_reward_per_timestep)

        with open(os.path.join('logs', 'log.txt'), mode) as f:
            log = (f"Agent incomes: {incomes}\n" +
                   f"Agent endowments: {endowments}\n" +
                   f"Agent Specialization coeff: {correlation}\n" +
                   f"Total productivity: {productivity}\n" +
                   f"Total equality: {equality}\n" + 
                   f"Total utility: {equality * productivity}\n" +
                   f"Income breakdown:\n{b[1]}\n\n" + 
                   f"Activity breakdown:\n{b[-1]}" 
                )
            f.write(log)

        plt.clf()
        fig, ax = plt.subplots()
        ax.set_xlabel('Timestep', fontsize=14)
        ax.set_ylabel('Cumulative Reward', fontsize=14)

        ax.plot(total_reward_per_timestep[0], label='Agent 0')
        ax.plot(total_reward_per_timestep[1], label='Agent 1')
        ax.plot(total_reward_per_timestep[2], label='Agent 2')
        ax.plot(total_reward_per_timestep[3], label='Agent 3')
        
        fig.tight_layout()
        plt.grid(False)

        ax.legend(loc='best', fontsize=14)
        filepath = os.path.join('logs', 'reward_graph.png')
        plt.savefig(filepath)
        plt.clf()
        plt.close(fig)