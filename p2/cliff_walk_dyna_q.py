# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import argparse

parser = argparse.ArgumentParser(description='Choose to train or load')
parser.add_argument('--mode', type=str, default='new', choices=['new', 'load'])
parser.add_argument('--path', type=str, default='./models/dynaq_agent.pkl')
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

##### START CODING HERE #####

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
agent = QLearningAgent(all_actions) 

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

            # agent learns from experience
            agent.learn(s, a, r, s_, isdone)

            s = s_
            episode_reward += r

            if isdone:
                time.sleep(0.1)
                break
            
        print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)  
    print('\ntraining over\n')   

    agent.save()

agent.gogogo()

# close the render window after training.
env.close()

##### END CODING HERE #####