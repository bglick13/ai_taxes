# import torch
import time

from torch import stack, zeros, clamp, min, save, load
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ai_economist import foundation
from ai_economist.foundation.scenarios.utils import social_metrics
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


def rollout(env_config, keys, constructors, state_dicts, n_rollouts, num_steps, _eval=False, device='cuda', completions=0):
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
    env._completions = completions  # Need this to make annealing work with multiprocessing
    obs = env.reset()
    t = range(n_rollouts)
    memory = dict((key, Memory()) for key in keys)
    models = dict((key, constructors[key](device=device)) for key in keys)
    for key, model in models.items():
        model.build_models(ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True))
        model.load_state_dict(state_dicts[key], strict=False)
    for rollout in t:
        obs = env.reset(force_dense_logging=_eval)
        states, actions, logprobs, rewards, values, done, hcs = (dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys),
                                                                 dict((key, []) for key in keys))
        model_hcs = dict((key, (zeros(1, 2 if 'p' in key else len(key), models[key].lstm_size, device=device),
                                zeros(1, 2 if 'p' in key else len(key), models[key].lstm_size, device=device))) for key in keys)
        metrics = dict(total_builds=0, total_immigrations=0)
        for step in range(num_steps):
            obs_batches = dict((key, ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True)) for key in keys)
            for key, item in model_hcs.items():
                hcs[key].append(item)
            action_dict = dict()
            for key in keys:
                dist, value, hc = models[key](obs_batches[key], model_hcs[key], det=False)
                hc = (hc[0].detach(), hc[1].detach())
                model_hcs[key] = hc
                a = dist.sample().detach()
                if 'p' not in key:
                    metrics['total_builds'] += (a == 1).detach().cpu().numpy().sum()
                    metrics['total_immigrations'] += (a >= 50).detach().cpu().numpy().sum()
                    actions[key].append(a.detach())
                    logprob = dist.log_prob(a).detach()
                    logprobs[key].append(logprob)
                    value = value.squeeze()
                    values[key].append(value)
                    action_dict.update(dict((i, _a.detach().cpu().numpy()) for i, _a in zip(obs_batches[key].order, a)))

                elif 'p' in key:
                    if step % env.world.planner.state['period'] == 0:
                        actions[key].append(a.detach())
                        logprob = dist.log_prob(a).detach()
                        logprobs[key].append(logprob)
                        value = value.squeeze()
                        values[key].append(value)
                        n_nations = len(obs_batches[key].order)
                        n_actions = a.shape[1] // n_nations
                        p_array = np.concatenate([_a.detach().cpu().numpy()[i * n_actions: (i + 1) * n_actions] for i, _a in enumerate(a)])
                        action_dict.update(dict(p=p_array))
                    else:
                        actions[key].append(actions[key][-1])
                        logprobs[key].append(logprobs[key][-1])
                        values[key].append(values[key][-1])

            next_obs, rew, is_done, info = env.step(action_dict)
            for key, obs in obs_batches.items():
                states[key].append(obs)
            for key in keys:
                if 'p' in key:
                    rew_dict = rew['p']
                    rewards[key].append(np.array(list(rew_dict.values())))
                else:
                    rewards[key].append(np.array([rew[k] for k in key]))
            for key in keys:
                done[key].append(is_done['__all__'])

            obs = next_obs

        obs_batches = dict((key, ObservationBatch(obs, key, flatten_action_masks=False if 'p' in key else True)) for key in keys)
        for key, obs in obs_batches.items():
            _, next_value, hc = models[key](obs, model_hcs[key])
            next_value = next_value.detach().cpu().numpy()
            values_k = stack(values[key]).detach().cpu().numpy()
            discounted_rewards = compute_gae(next_value, rewards[key], done[key], values_k)
            advantage = (discounted_rewards - values_k).tolist()
            discounted_rewards = discounted_rewards.tolist()
            memory[key].add_trace(states[key], actions[key], logprobs[key], discounted_rewards, advantage, hcs[key])
    if _eval:
        return memory, env.previous_episode_dense_log, env.mixing_weight_gini_vs_coin
    else:
        return memory, metrics


def compute_gae(next_value, rewards, masks, values, tau=0.98, gamma=0.998):
    values = np.vstack((values, next_value.T))
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        try:
            delta = rewards[step] + (gamma * values[step + 1] * masks[step]).squeeze() - values[step]
        except TypeError:
            print('here')
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

    def load_weights_from_file(self, from_checkpoint=False):
        if from_checkpoint:
            files = os.listdir('weights/temp')
            files.sort(key=lambda x: os.path.getmtime(f'weights/temp/{x}'))
            checkpoint = load(f'weights/temp/{files[-1]}')
        else:
            checkpoint = load(f'weights/final.torch')
        for key, value in self.models.items():
            # weights = load(os.path.join('weights', str(key)+'_final.torch'))
            weights = checkpoint[f'model_{key}']
            opt = checkpoint[f'opt_{key}']
            self.models[key].load_state_dict(weights)
            self.optimizers[key].load_state_dict(opt)

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
                if 'p' in key:
                    actions = actions.reshape(-1, actions.shape[-1]).to('cuda')
                else:
                    actions = actions.reshape(-1, 1).to('cuda')
                old_logprobs = old_logprobs.flatten().unsqueeze(-1).to('cuda')
                advantages = stack(advantages).reshape(-1, 1).to('cuda')
                rewards = stack(rewards).reshape(-1, 1).to('cuda')
                hs = hcs[0].squeeze()
                cs = hcs[1].squeeze()
                hs = hs.reshape(hs.shape[0] * hs.shape[1], hs.shape[2]).to('cuda').unsqueeze(0)  # (batch_size, n_agents, hidden_size)
                cs = cs.reshape(cs.shape[0] * cs.shape[1], cs.shape[2]).to('cuda').unsqueeze(0)  # (batch_size, n_agents, hidden_size)

                hcs = (hs, cs)

                obs_batch = ObservationBatch([world_maps.float(), flat_inputs.float()])
                dist, value, hc = self.models[key](obs_batch, hcs)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions)
                if 'p' in key:
                    new_log_probs = new_log_probs.flatten().unsqueeze(-1)
                ratio = (new_log_probs - old_logprobs).exp()
                if 'p' in key:
                    advantages = advantages.repeat(1, actions.shape[-1]).reshape(-1, 1)
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
        rollouts_per_jobs = 1
        n_rollouts_per_training_step = 12
        steps_per_rollouts = 1000

        def completion_number(it, job_idx):
            return it * n_rollouts_per_training_step * rollouts_per_jobs + job_idx

        self.env_config['dense_log_frequency'] = None
        writer = SummaryWriter()
        t = trange(n_training_steps, desc='Training Iteration', leave=True)
        if not os.path.isdir(f'weights/temp'):
            os.makedirs(f'weights/temp')

        env = foundation.make_env_instance(**self.env_config)
        obs = env.reset()
        agent = env.get_agent('0')
        for i in range(agent.action_spaces):
            if i == 0:
                print(f'{i}: NO-OP')
                continue
            action_name, action = agent.single_action_map.get(i)
            print(f'{i}: {action_name}: {action}')

        for it in t:
            state_dicts = dict((k, v.to('cpu').state_dict()) for k, v in self.models.items())

            start = time.time()
            with mp.Pool(n_jobs) as pool:
                result = pool.starmap(rollout, [(self.env_config, np.array(key_order)[:, 0],
                                                 self.model_key_to_constructor, state_dicts, rollouts_per_jobs,
                                                 steps_per_rollouts, False, self.device, completion_number(it, _))
                                                for _ in range(n_rollouts_per_training_step)])
            writer.add_scalar(f'Metric/n_builds', np.mean([r[1]['total_builds'] for r in result]), it)
            writer.add_scalar(f'Metric/n_immigrations', np.mean([r[1]['total_immigrations'] for r in result]), it)
            print(f'rollouts took {time.time() - start}s')
            for key in self.model_key_to_constructor.keys():
                self.memory[key] = join_memories([r[0][key] for r in result])

            checkpoint = dict()
            for key in self.memory.keys():
                if 'p' in key:
                    epochs = 4
                else:
                    epochs = 16
                batch_size = 3000 // len(key)
                losses, all_rewards = self.update(key, epochs, batch_size)
                writer.add_scalar(f'Loss/train_{key}', np.mean(losses), it)
                writer.add_scalar(f'Reward/train_{key}', stack(all_rewards).mean().detach().cpu().numpy().round(3), it)
                checkpoint[f'model_{key}'] = self.models[key].state_dict()
                checkpoint[f'opt_{key}'] = self.optimizers[key].state_dict()

                self.memory[key].clear_memory()
            save(checkpoint, f'weights/temp/checkpoint_{it}.torch.temp')
            t.refresh()

        checkpoint = dict()
        for key, spec in key_order:
            checkpoint[f'model_{key}'] = self.models[key].state_dict()
            checkpoint[f'opt_{key}'] = self.optimizers[key].state_dict()
        save(checkpoint, f'weights/final.torch')

    def eval(self, key_order: List[Tuple[Tuple, Dict]], mode='w+'):
        if not os.path.exists('logs'):
            os.mkdir('logs')

        self.env_config['components'][-1][1]['tax_annealing_schedule'] = None
        self.env_config['mixing_weight_gini_vs_coin'] = dict(foo_land=0.8, bar_land=0.2)
        dense_logs = list()
        state_dicts = dict((k, v.to('cpu').state_dict()) for k, v in self.models.items())

        _, log, mixing_weights = rollout(self.env_config, np.array(list(key_order.keys())), self.model_key_to_constructor, state_dicts,
                n_rollouts=1, num_steps=1000, _eval=True)
        with open('logs/dense_log.pickle', 'wb') as f:
            pickle.dump(log, f)
        with open('logs/mixing_weights.pickle', 'wb') as f:
            pickle.dump(mixing_weights, f)
        dense_logs.append(log)
        b = breakdown(log)

        incomes = list(np.round(b[1]['Total'], 3))
        percent_income_from_build = list(np.round(b[1]['Build']/b[1]['Total'], 3))
        endowments = list(b[2])
        productivity = sum(incomes)
        equality = social_metrics.get_equality(np.array(endowments))

        action_counts = {str(idx):{'Gather':0, 'Build':0} for idx in range(len(incomes))}
        for i in range(len(log['actions'])):
            action_vector = log['actions'][i]
            for idx in action_counts.keys():
                if 'Gather' in action_vector[idx].keys():
                    action_counts[idx]['Gather'] += 1
                if 'Build' in action_vector[idx].keys():
                    action_counts[idx]['Build'] += 1
        specializations = [action_counts[idx]['Build'] / (action_counts[idx]['Build'] + action_counts[idx]['Gather'])
            for idx in action_counts.keys()]

        tax_schedules = dict((i,log['PeriodicTax'][i])
            for i in range(len(log['PeriodicTax'])) if isinstance(log['PeriodicTax'][i],dict))
        immigration = dict((i,log['Citizenship'][i][0])
            for i in range(len(log['Citizenship'])) if len(log['Citizenship'][i]) > 0)
        population_delta = {nation:[] for nation in tax_schedules[99]['schedule'].keys()}
        for i in range(len(log['Citizenship'])):
            if i in immigration.keys():
                for nation in population_delta.keys():
                    delta = 0
                    for immigrant in immigration[i]:
                        if nation == immigrant['from_nation']:
                            delta = delta - 1
                        elif nation == immigrant['to_nation']:
                            delta = delta + 1
                    population_delta[nation].append(delta)
            else:
                for nation in population_delta.keys():
                    population_delta[nation].append(0)

        rewards = [[log['rewards'][i][agent] for i in range(len(log['rewards']))] for agent in log['rewards'][0].keys()]
        # TODO(Connor): this is hard coded for just 4 agents
        total_reward_per_timestep = [[rewards[0][0]],[rewards[1][0]],[rewards[2][0]],[rewards[3][0]]]
        for i in range(1, len(rewards[0])):
            for j in range(len(rewards)-1):
                total_reward_per_timestep[j].append(rewards[j][i] + total_reward_per_timestep[j][i-1])
        total_reward_per_timestep = np.array(total_reward_per_timestep)

        with open(os.path.join('logs', 'log.txt'), mode) as f:
            log = (f"Agent incomes: {incomes}\n" +
                   f"Agent endowments: {endowments}\n" +
                   f"Agent specializations: {specializations}\n" +
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