


import warnings
import os 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pygame as pg
import gymnasium as gym
warnings.filterwarnings("ignore")
pg.init()

env = gym.make("LunarLanderContinuous-v3", render_mode="human")
initial_observation = env.reset()[0]

print(f"Initial Observation: {initial_observation}")




'''
print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("Action space low:", env.action_space.low)
print("Action space high:", env.action_space.high)
print("Sample random action:", env.action_space.sample())
'''


# class ActorNetwork()




runs = 5

for run in range(0,runs):
    reward_list = []
    running = True
    while running:    
        events = pg.event.get()
        
        for event in events:
            if event.type == 256:
                env.close()
                running = False
        
        action = env.action_space.sample()
        action_calculations = env.step(action)
        
        observation = action_calculations[0]
        reward = action_calculations[1]
        terminated = action_calculations[2]
        truncated = action_calculations[3]
        info = action_calculations[4]
        
        reward_list.append(float(reward))
        
        # print("=" * 10)
        # print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")
        
        if terminated == True:
                env.reset()
                running = False
        
        # print(f"Info: {info}")
        
        # sleep for framerate


    # print(reward_list)
    print(np.sum(reward_list))
    # print(f"Length of reward_list: {len(reward_list)}")
