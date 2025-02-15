import gymnasium as gym
import matplotlib.pyplot as plt
import time # for benchmarking
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections # For dequeue for the memory buffer
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter=0 # num of transitions in the memory
                             # this count is required to delay learning
                             # until the buffer is sensibly full
        self.index=0         # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        self.memory.save(state, action, reward, new_state, done)  

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  
        
        
        
    
class QAgent(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.q_func = QNN(8, 4, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)

    
    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        return np.argmax(action_values.cpu().data.numpy())
 
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Update the target values
        q_next = self.q_func(new_states).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states).gather(1, actions)
        
        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
        # 4. Reduce the exploration rate
        self.reduce_epsilon()
    

if __name__ == "__main__":
    LEARN_EVERY = 4
    def train_agent(n_episodes=2000):
        print(f"Training a DQN agent on {n_episodes} episodes.")
        env = gym.make("LunarLander-v3")
        agent = QAgent(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.001, mem_size=200000, batch_size=128, epsilon_end=0.01)
            
        scores = []
        start = time.time()
        for i in range(n_episodes):
            terminated = False
            truncated = False
            score = 0
            state = env.reset()[0]
            steps = 0
            while not (terminated or truncated):
                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.save(state, action, reward, new_state, terminated)
                state = new_state
                if steps > 0 and steps % LEARN_EVERY == 0:
                    agent.learn()
                steps += 1
                score += reward
                
            scores.append(score)
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            if (i+1) % 10 == 0 and i > 0:
                # Report expected time to finish the training
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        return agent, scores



    def test_agent(agent, n_episodes=200):
        print(f"Testing a DQN agent on {n_episodes} episodes.")
        env = gym.make("LunarLander-v2")

        scores = []
        start = time.time()
        for i in range(n_episodes):
            terminated = False
            truncated = False
            score = 0
            state = env.reset()[0]
            while not (terminated or truncated):
                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                state = new_state
                score += reward
                
            scores.append(score)
            avg_score = np.mean(scores[max(0, i-100):(i+1)])

            if (i+1) % 10 == 0 and i > 0:
                # Report expected time to finish the training
                print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                        (time.time() - start)/60, 
                                                                                                                        n_episodes, 
                                                                                                                        (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                        score, 
                                                                                                                        avg_score))
                    
        return scores


    # Uncomment to train
    agent, ed_scores = train_agent(n_episodes=1200)
    test_scores = test_agent(agent)
    ed_indexes = np.arange(len(ed_scores)).tolist()
    test_indexes = np.arange(len(test_scores)).tolist()

    plt.plot(ed_indexes, ed_scores)
    plt.xlabel("Эпизоды обучения")
    plt.ylabel("Награды")
    plt.title("Результаты работы агента в процессе обучения")
    plt.show()

    plt.plot(test_indexes, test_scores)
    plt.xlabel("Эпизоды тестирования")
    plt.ylabel("Награды")
    plt.title("Результаты работы агента в процессе тестирования")
    plt.show()
