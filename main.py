


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

framerate = 600

runs = 300
runs_to_render = [0,3,4]
gamma = 0.99

training_env = gym.make("LunarLanderContinuous-v3", render_mode="human")
render_env = gym.make("LunarLanderContinuous-v3", render_mode=None)
initial_observation = training_env.reset()[0]
state = T.from_numpy(training_env.reset()[0])

training_env.metadata["render_fps"] = framerate
render_env.metadata["render_fps"] = framerate

# print(f"Initial Observation: {initial_observation}")

'''
print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("Action space low:", env.action_space.low)
print("Action space high:", env.action_space.high)
print("Sample random action:", env.action_space.sample())
'''
class actor_network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size,256)
        self.layer2 = nn.Linear(256,output_size)

    def forward(self, state: T.tensor):
        action = T.tanh(self.layer2(F.relu(self.layer1(state)))) #the F.relu and T.tanh is the hyperbolic tangent to make this network non-linear, and to bound the tensor to values -1 to 1.
        return action

class critic_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        input_dim = state_dim + action_dim
        self.layer1 = nn.Linear(input_dim,256)
        self.layer2 = nn.Linear(256,1) #outputs a single q value
        
    def forward(self,state,action):
        combined = T.cat([state,action], dim=-1)
        q_value = self.layer2(F.relu(self.layer1(combined)))
        return q_value
        

lunar_actor = actor_network(8,2)
lunar_critic = critic_network(8,2)

actor_optimizer = optim.Adam(lunar_actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(lunar_critic.parameters(), lr=0.001)

total_reward_for_alls_runs = []

# Game loop

for run in range(0,runs):
    
    train_obs, _ = training_env.reset()
    render_obs, _ = render_env.reset()
    state = T.from_numpy(train_obs)
    
    reward_list_for_run = []
    running = True
    print(f"Run {run}")
    
    while running:    
        events = pg.event.get()
        
        for event in events:
            if event.type == 256:
                training_env.close()
                render_env.close()
                print("=" * 50)
                print(f"total_reward_for_alls_runs: {total_reward_for_alls_runs}")
                running = False
        # print("=" *10)
        # print(f"Run {run}")
        
        if (run in runs_to_render) and render_env.render_mode == "human":
            render_env.render()
        
        action = lunar_actor(state)
        action_calculations = training_env.step(action.detach().numpy())
        
        # print(f"Action: {action}")
        
        next_state = T.from_numpy(action_calculations[0])
        reward = action_calculations[1]
        terminated = action_calculations[2]
        truncated = action_calculations[3]
        info = action_calculations[4]
        
        q_value_for_actor = lunar_critic(state,action).mean()

        actor_loss = -q_value_for_actor
        # print(f"actor_loss: {actor_loss[0]}")
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        q_value_for_critic = lunar_critic(state,action.detach())
        with T.no_grad():
            next_action = lunar_actor(next_state)
            target = reward + gamma * lunar_critic(next_state,next_action) #bellman
        critic_loss = F.mse_loss(q_value_for_critic,target.detach())
        # print(f"Critic Loss: {critic_loss[0]}")
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        state = next_state
        
        reward_list_for_run.append(float(reward))
        
        # print("=" * 10)
        # print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")
        
        if terminated == True:
            if "success" in info:
                print("SUCCESS")
            else:
                print("FAILURE")
            training_env.reset()
            render_env.reset()
            running = False
        
        total_reward_for_one_run = float(np.sum(reward_list_for_run))
        
    
    print(f"total_reward_for_one_run: {total_reward_for_one_run}")
    total_reward_for_alls_runs.append(total_reward_for_one_run)

