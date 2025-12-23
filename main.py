


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
# Try this and observe what happens

env = gym.make("LunarLanderContinuous-v3", render_mode="human")
env.reset()


print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("Action space low:", env.action_space.low)
print("Action space high:", env.action_space.high)
print("Sample random action:", env.action_space.sample())

running = True
while running:    
    events = pg.event.get()
    
    for event in events:
        if event.type == 256:
            env.close()
            running = False
    
    action = env.action_space.sample()
    action_calculations = env.step(action)
    
    # sleep for framerate
    

    
    
    
