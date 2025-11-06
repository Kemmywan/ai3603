# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import SarsaAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

import argparse

parser = argparse.ArgumentParser(description='Choose to train or load')
parser.add_argument('--mode', type=str, default='new', choices=['new', 'load'])
parser.add_argument('--path', type=str, default='./models/sarsa_agent.pkl')
args = parser.parse_args()

##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

####### START CODING HERE #######

# construct the intelligent agent.
agent = SarsaAgent(all_actions) 

if args.mode == 'load':
    if os.path.exists(args.path):
        agent.load(args.path)
    else:
        raise FileNotFoundError(f"Model not found in {args.path}")
else:
    # start training
    for episode in range(1000):
        # record the reward in an episode
        episode_reward = 0
        # reset env
        s = env.reset()
        # render env. You can remove all render() to turn off the GUI to accelerate training.
        # env.render()
        # agent interacts with the environment
        for iter in range(500):
            # choose an action
            a = agent.choose_action(s)
            s_, r, isdone, info = env.step(a)
            # env.render()
            # update the episode reward
            print(f"{s} {a} {s_} {r} {isdone}")

            a_ = agent.choose_action(s_)
            # agent learns from experience
            agent.learn(s, a, r, s_, a_, isdone)

            
            s = s_
            a = a_
            episode_reward += r

            if isdone:
                time.sleep(0.1)
                break
        print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
    print('\ntraining over\n')   

    agent.save()

agent.gogogo()

# state = env.reset()

# env.render()

# action = np.argmax(agent.q_table[state])

# print(agent.q_table[13])

# ---------------------------------------------------------

# state = env.reset()

# env.render()

# path = [state]
# total_reward = 0

# for step in range(100):

#     sorted_actions = np.argsort(agent.q_table[state])[::-1]

#     for action in sorted_actions:
#         print(action)
#         print(path)
#         sim_state = state + (-12 if action == 0 else 0) + (1 if action == 1 else 0) + (12 if action == 2 else 0) + (-1 if action == 3 else 0) 
#         if sim_state in path or sim_state < 0 or sim_state > 36 and sim_state < 47:
#             continue
#         else:
#             break
        
#     next_state, reward, isdone, info = env.step(action)

#     env.render()

#     time.sleep(0.5)

#     total_reward += reward

#     path.append(next_state)

#     print(f"--> go to ({next_state // 12},{next_state % 12}) with {reward} added into total reward: {total_reward}")

#     if isdone:
#         print("Arrive!")
#         break

#     state = next_state

# close the render window after training.
env.close()

####### END CODING HERE #######


