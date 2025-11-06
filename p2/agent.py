# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
import pickle
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class BaseAgent(object):
    """懒惰所以定义一个基类用于继承"""
    def __init__(self, all_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # CliffWaking_Status_Space=4*12=48
        self.q_table = np.zeros((48, len(all_actions)))

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            action = np.argmax(self.q_table[observation])
        return action
    
    def save(self, filepath="./models/sarsa_agent.pkl"):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.lr = data['lr']
        self.gamma = data['gamma']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_min = data['epsilon_min']

    def gogogo(self):
        "render the route onboard according to the q_table"
        env = gym.make("CliffWalking-v0")
        
        RANDOM_SEED = 0
        env.seed(RANDOM_SEED)

        state = env.reset()

        env.render()

        path = [state]
        total_reward = 0

        for step in range(100):

            sorted_actions = np.argsort(self.q_table[state])[::-1]

            # Make a valid and q-smallest action
            for action in sorted_actions:
                print(action)
                print(path)
                sim_state = state + (-12 if action == 0 else 0) + (1 if action == 1 else 0) + (12 if action == 2 else 0) + (-1 if action == 3 else 0) 
                if sim_state in path or sim_state < 0 or sim_state > 36 and sim_state < 47:
                    continue
                else:
                    break
                
            next_state, reward, isdone, info = env.step(action)

            env.render()

            time.sleep(0.5)

            total_reward += reward

            path.append(next_state)

            print(f"--> go to ({next_state // 12},{next_state % 12}) with {reward} added into total reward: {total_reward}")

            if isdone:
                print("Arrive!")
                break

            state = next_state
        
        env.close()

        return None

class SarsaAgent(BaseAgent):
    ##### START CODING HERE #####
    def __init__(self, all_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """initialize the agent. Maybe more function inputs are needed."""
        super().__init__(all_actions, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)
    
    def learn(self, state, action, reward, next_state, next_action, isdone):
        """learn from experience
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        # Q for now
        current_q = self.q_table[state, action]

        if isdone:
            target_q = reward
        else:
            target_q = reward + self.gamma * self.q_table[next_state, next_action]

        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        if isdone and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ##### END CODING HERE #####

class QLearningAgent(BaseAgent):
    ##### START CODING HERE #####
    def __init__(self, all_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """initialize the agent. Maybe more function inputs are needed."""
        super().__init__(all_actions, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)


    def learn(self, state, action, reward, next_state, isdone):
        """learn from experience
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        # Q for now
        current_q = self.q_table[state, action]

        if isdone:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
            # Use the max q from next_state 

        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        if isdone and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    ##### END CODING HERE #####
    
class Dyna_QAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, plan_steps=5):
        """initialize the agent. Maybe more function inputs are needed."""
        super().__init__(all_actions, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)

        self.plan_steps = plan_steps

        self.model = {}

        self.history = []
    
    def learn(self, state, action, reward, next_state, isdone):
        """learn from experience
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        # Q for now
        current_q = self.q_table[state, action]

        if isdone:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
            # Just like qlearning

        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        self.model[(state, action)] = (reward, next_state, isdone)
        
        if (state, action) not in self.history:
            self.history.append((state, action))
        
        for _ in range(self.plan_steps):
            if self.history == []:
                break

            # Select experience randomly from history
            s, a = self.history[np.random.randint(len(self.history))]

            r, s_, d = self.model[(s, a)]

            q_sim = self.q_table[s, a]

            if d:
                t_q_sim = r
            else:
                t_q_sim = r + self.gamma * np.max(self.q_table[s_])
            
            self.q_table[s, a] = q_sim + self.lr * (t_q_sim - q_sim)

        if isdone and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath="./models/dynaq_agent.pkl"):
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'plan_steps': self.plan_steps,
            'model': self.model,
            'history': self.history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.lr = data['lr']
        self.gamma = data['gamma']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_min = data['epsilon_min']
        self.plan_steps = data['plan_steps']
        self.model = data['model']
        self.history = data['history']

    ##### END CODING HERE #####
