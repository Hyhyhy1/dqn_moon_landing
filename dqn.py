import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
import collections
from utils.qnet import *
    

class DQN():
    def __init__(self, action_n, model, trajectory_count, memory_size = 10000, lr=0.001, epsilon=1, gamma = 0.95, batch_size=64):
        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_n = action_n
        self.batch_size = batch_size
        self.epsilon_decrease = 1/trajectory_count
        self.memory = collections.deque(maxlen=memory_size)
        self.optim = torch.optim.Adam(self.model.parameters(), lr)


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


    def get_action(self, state):
        #print(state)
        temp = self.model(torch.from_numpy(state))
        qvalues = temp.detach().numpy()
        
        prob = self.get_action_prob(qvalues)
        #print(prob)
        action = np.random.choice(np.arange(self.action_n), p=prob)

        return action



    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])

        if len(self.memory) < self.batch_size * 10:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, dones, next_states = self.parse_batch(batch)

        qvalues = self.model(states)
        next_qvalues = self.model(next_states)

        targets = qvalues.clone()

        for i in range(len(batch)):
            targets[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_qvalues[i])

        loss = torch.mean((targets.detach() - qvalues) ** 2)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.epsilon = max(0, self.epsilon - self.epsilon_decrease)


if __name__ == '__main__':

    #env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("LunarLander-v2")

    state_dim = env.observation_space.shape[0]
    action_n = env.action_space.n

    trajectory_count = 1000
    trajectory_len = 400
    model = Qnet(state_dim, action_n, 128)
    agent = DQN(action_n, model, trajectory_count)
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

        if i%100 == 0:
            print(f"Trajectory №{i}: Reward: {np.mean(rewards)}")
            rewards.clear()

    

    env = gym.make("LunarLander-v2", render_mode="human")
    state = env.reset()[0]
    for _ in range(trajectory_len):

        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.fit(state, action, reward, done, next_state)
        state = next_state

        env.render()

        if done: 
            break