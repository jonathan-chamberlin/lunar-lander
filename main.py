


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

env = gym.make("LunarLander-v3", render_mode="human")
env.reset()
running = True
while running:    
    events = pg.event.get()
    for event in events:
        if event.type == 256:
            env.close()
            running = False
    # print(f"Event type: {event.type}")
    # print(f"Event object: {event}")
    
    
    
    
