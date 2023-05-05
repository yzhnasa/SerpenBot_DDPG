import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
from datetime import datetime
from actor_critic import Actor, Critic
from utilities import OUnoise, ExperienceMemory

SEED = random.seed(datetime.now())
HIDDEN_UNITES = 50
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
WEIGEHT_DECAY_RATE = 0
BATCH_SIZE = 128
GAMMA = 0.98
TAU = 1e-3
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DDPG(object):
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.memory = ExperienceMemory(action_dim, state_dim)
        if os.path.isfile('actor.pkl'):
            self.load_actor()
        else:
            self.actor_online = Actor(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.actor_target = Actor(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.actor_online.double()
            self.actor_target.double()
        self.actor_optimizer = optim.Adam(self.actor_online.parameters(), lr=ACTOR_LEARNING_RATE)
        if os.path.isfile('critic.pkl'):
            self.load_critic()
        else:
            self.critic_online = Critic(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.critic_target = Critic(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.critic_online.double()
            self.critic_target.double()
        self.critic_optimizer = optim.Adam(self.critic_online.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=WEIGEHT_DECAY_RATE)
        self.noise = OUnoise(self.action_dim, SEED)

    def select_action(self, current_state, add_noise=True):
        current_state = torch.from_numpy(current_state)#.float().to(DEVICE)
        self.actor_online.eval()
        with torch.no_grad():
            action = self.actor_online.forward(current_state.unsqueeze(0))#.cpu().data.numpy()
        if add_noise:
            action += self.noise.sample_noise()

        action = np.clip(action, 0, 1)
        self.actor_online.train()
        return action

    def is_memory_full(self):
        return self.memory.is_memory_full()

    def learn(self):
        current_states, actions, rewards, next_states = self.memory.get_experiences(BATCH_SIZE)
        # Critic Update
        next_actions = self.actor_target(next_states)
        with torch.no_grad():
            q_reward_targets_next = self.critic_target(next_states, next_actions)
        q_reward_targets = rewards + GAMMA * q_reward_targets_next
        q_reward_expected = self.critic_online(current_states, actions)
        critic_loss = F.mse_loss(q_reward_targets, q_reward_expected) # TD error
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_online.parameters(), 1)
        self.critic_optimizer.step()
        # Actor update
        action_predictions = self.actor_online(current_states)
        actor_loss = -self.critic_online(current_states, action_predictions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Soft update
        self.soft_update(self.critic_target, self.critic_online)
        self.soft_update(self.actor_target, self.actor_online)

    def soft_update(self, target, local):
        for target_params, local_params in zip(target.parameters(), local.parameters()):
            target_params.data.copy_(TAU * local_params.data + (1 - TAU) * target_params.data)

    def store_experience(self, current_state, action, reward, next_state):
        self.memory.store_experience(current_state, action, reward, next_state)

    def save_actor(self):
        torch.save(self.actor_online, 'actor.pkl')

    def load_actor(self):
        self.actor_online = torch.load('actor.pkl').double()
        self.actor_target = torch.load('actor.pkl').double()

    def save_critic(self):
        torch.save(self.critic_online, 'critic.pkl')

    def load_critic(self):
        self.critic_online = torch.load('critic.pkl').double()
        self.critic_target = torch.load('critic.pkl').double()

