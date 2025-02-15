import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import collections
import matplotlib.pyplot as plt
from utils.qnet import *

class DDQN(nn.Module):
    def __init__(self, state_dim, action_n, hiden_dim, trajectory_count,
                  steps_before_update=500, memory_size=500000, batch_size=128, gamma=0.99, epsilon = 0.99, lr=0.001, tau= 0.01):
        super(DDQN, self).__init__()
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_n = action_n
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon_decrease = 1/trajectory_count
        self.memory = collections.deque(maxlen=self.memory_size)

        self.counter = 0
        self.steps_before_update = steps_before_update

        self.q_func = Qnet(state_dim, action_n, hiden_dim, 49)
        self.q_func_target = Qnet(state_dim, action_n, hiden_dim, 49)

        self.optim = torch.optim.Adam(self.q_func.parameters(), lr=self.lr)


    def fit(self, state, action, reward, done, next_state):

        #1 Подготовка батча
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) < self.batch_size * 2:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = self.parse_batch(batch)

        #2 Обновление целевых значений
        q_current = self.q_func(states)
        q_next = self.q_func_target(next_states)

        targets = q_next.clone()

        for i in range(len(batch)):
            targets[i][actions[i]] = rewards[i] + (1-dones[i]) * self.gamma * torch.argmax(q_next[i])
        
        #3 Обновление основной сети
        loss = torch.nn.functional.mse_loss(input=q_current, target=targets)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        #4 Обновление целевой сети
        #if self.counter % self.steps_before_update == 0:
        #    for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
        #        target_param.data.copy_(local_param.data)

        if self.counter % self.steps_before_update == 0:
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

        self.epsilon = max(0, self.epsilon - self.epsilon_decrease)
        self.counter+=1


    def get_action(self, state):
        #print(state)
        temp = self.q_func(torch.from_numpy(state))
        qvalues = temp.detach().numpy()
        
        prob = self.get_action_prob(qvalues)
        #print(prob)
        action = np.random.choice(np.arange(self.action_n), p=prob)

        return action


    def get_action_prob(self, q_values):
        prob = np.ones(self.action_n) * self.epsilon / self.action_n
        argmax_action = np.argmax(q_values)
        prob[argmax_action] += 1 - self.epsilon
        return prob


    def parse_batch(self, batch):
        states, actions, rewards, dones, next_states = [],[],[],[],[]
        for i in range(len(batch)):
            states.append(batch[i][0])
            actions.append(batch[i][1])
            rewards.append(batch[i][2])
            dones.append(batch[i][3])
            next_states.append(batch[i][4])
        states = torch.from_numpy(np.array(states))
        next_states = torch.from_numpy(np.array(next_states))
        return states, actions, rewards, dones, next_states


if __name__ == '__main__':

    #env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n
    hiden_dim = 128

    trajectory_count = 1000
    trajectory_len = 400
    agent = DDQN(state_dim, action_n, hiden_dim, trajectory_count,lr=0.001, tau=0.1)
    rewards = []
    for i in range(trajectory_count+1):

        total_reward = 0
        state = env.reset()[0]

        for _ in range(trajectory_len):

            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            total_reward+=reward

            agent.fit(state, action, reward, done, next_state)
            state = next_state

            if done: 
                break
        
        rewards.append(total_reward)

        if i%10 == 0:
            print(f"Trajectory №{i}: Reward: {np.mean(rewards)}")
            rewards.clear()

    input()

    for i in range(8):

        total_reward = 0
        state = env.reset()[0]

        for _ in range(trajectory_len):

            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            total_reward+=reward

            agent.fit(state, action, reward, done, next_state)
            state = next_state

            if done: 
                break
        
        print(f"Trajectory №{i}: Reward: {total_reward}")
