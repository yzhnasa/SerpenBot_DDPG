#!/usr/bin/python
import random
import numpy as np
import copy
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class OUnoise():
    """
    Ornstein–Uhlenbeck process noise, see https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, action_size, seed, mu=0.0, theta=0.03, sigma=0.07):
        """ Initialize OU noise.
        Params
        ======
            action_size: 	dimension of each action
            seed:			seed
            mu:				mean
            theta:			mean reversion rate
            sigma:			std
        """
        self.mu = np.ones(action_size) * mu
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()

    def sample_noise(self):
        ns = self.state
        dn = self.theta * (self.mu - ns) + self.sigma * np.array([np.random.randn() for i in range(len(ns))])
        self.state = ns + dn
        return self.state

    def reset(self):
        self.state = copy.copy(self.mu)

MEMORY_CAPACITY = 5000

class ExperienceMemory(object):
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.current_index = 0
        self.total_stored = 0
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)

    def store_experience(self, current_state, action, reward, next_state):
        experience = np.hstack((current_state, action, reward, next_state))
        self.memory[self.current_index, :] = experience
        # print(self.memory)
        self.current_index = (self.current_index + 1) % MEMORY_CAPACITY
        self.total_stored = self.total_stored + 1


    def get_experiences(self, batch_size):
        indices = np.random.choice(MEMORY_CAPACITY, size=batch_size)
        experiences = self.memory[indices, :]
        current_states = experiences[:, :self.state_dim]
        actions = experiences[:, self.state_dim:self.state_dim+self.action_dim]
        rewards = experiences[:, -self.state_dim-1:-self.state_dim]
        next_states = experiences[:, -self.state_dim:]

        current_states = torch.from_numpy(current_states).double()#.float().to(DEVICE)
        actions = torch.from_numpy(actions).double()  # .float().to(DEVICE)
        rewards = torch.from_numpy(rewards).double()  # .float().to(DEVICE)
        next_states = torch.from_numpy(next_states).double()  # .float().to(DEVICE)
        return current_states, actions, rewards, next_states

    def is_memory_full(self):
        if self.total_stored > MEMORY_CAPACITY:
            return True
        return False