from typing import final
from gym import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from storage import RolloutStorage


class Runner:

    def __init__(self, net, env, num_envs, n_stack, rollout_size=5, num_updates=2500000, max_grad_norm=0.5,
    value_coeff=0.5, entropy_coeff=0.2, tensorboard_log=True, log_path='/home/tims/work/myprojects/exemples/AC/A2C/a2c/log', is_cuda=True, seed=42) -> None:
        
        #consts
        self.num_envs = num_envs
        self.n_stack = n_stack
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self.is_cuda = is_cuda
        #objects
        """Tensorboard logger"""
        self.writer = SummaryWriter(
            comment='statistics', 
            log_dir=log_path
            ) if tensorboard_log else None
        
        """Env"""
        self.env = env
        """Storege"""
        self.storage = RolloutStorage(
            rollout_size=self.rollout_size,
            num_envs=self.num_envs,
            # frame_shape=self.env.observation_space.shape[0:-1],
            frame_shape=[65,84],
            n_stack=self.n_stack,
            cuda=self.is_cuda,
            value_coeff=value_coeff,
            entropy_coeff=entropy_coeff,
            writer=self.writer
            )
        """Net"""
        self.net = net
        self.net.a2c.writer = self.writer
        if self.is_cuda:
            self.net = self.net.cuda()
        # self.writer.add_graph(self.net, input_to_model=(self.storage.states[0],)) --> not working for LSTMCEll

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.rollout_size):
            """Interact with the envs"""
            #call a2c
            a_t, log_p_a_t, entropy, value, a2c_features = self.net.a2c.get_action(self.storage.get_state(step))
            #accumulate episode entropy
            episode_entropy += entropy
            #interact 
            obs, rewards, dones, infos = self.env.step(a_t.cpu().numpy())
            #save episode reward
            self.storage.log_episode_rewards(infos)

            self.storage.insert(step, obs, a_t, rewards, log_p_a_t, value, dones)
            self.net.a2c.reset_reccurent_buffers(reset_indices=dones)
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))
        return final_value, episode_entropy 

    def train(self):
        """Env reset"""
        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))
        best_score = np.inf

        for num_update in range(self.num_updates):
            final_value, entropy = self.episode_rollout()
            self.net.optimizer.zero_grad()
            """Assemble loss"""
            loss = self.storage.a2c_loss(final_value, entropy)
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            if self.writer is not None:
                self.writer.add_scalar("loss", loss.item())
            self.net.optimizer.step()
            self.net.scheduler.step()
            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            if num_update % 10 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                new_score = self.storage.print_reward_stats()
            if len(self.storage.episode_rewards) > 1:
                new_score = np.mean(self.storage.episode_rewards)
                if new_score >= best_score:
                    best_score = new_score
                    print("model saved with best score: ", best_score, " at update #", num_update)
                    torch.save(self.net.state_dict(), "a2c_best_score")

            if num_update % 100 == 0:
                torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

            if self.writer is not None and len(self.storage.episode_rewards) > 1:
                self.writer.add_histogram("episode_rewards", torch.tensor(self.storage.episode_rewards))

        self.env.close()

