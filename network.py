from inputs import *
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class actor_network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 400)
        self.layer2 = nn.Linear(400, 300)
        # Separate output layers for each action dimension
        self.main_engine_layer = nn.Linear(300, 1)  # Main engine [0, 1]
        self.side_engine_layer = nn.Linear(300, 1)  # Side engine [-1, 1]

        # Initialize main engine bias to 0 so sigmoid(0) = 0.5 (moderate thrust)
        nn.init.constant_(self.main_engine_layer.bias, 0.0)

    def forward(self, state: T.tensor):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        # Apply appropriate activation for each action
        main_engine = T.sigmoid(self.main_engine_layer(x))  # [0, 1] for main thruster
        side_engine = T.tanh(self.side_engine_layer(x))      # [-1, 1] for side thruster

        # Concatenate to form action vector
        action = T.cat([main_engine, side_engine], dim=-1)
        return action
    
class OUActionNoise():
    def __init__(self, mu,sigma,theta,dt,x0,action_dimensions):
        self.sigma = sigma
        self.theta=theta
        self.dt=dt
        self.x0 = x0
        self.action_dimensions = action_dimensions
        
        if isinstance(mu, int):
            self.mu = np.array(mu*np.ones(action_dimensions))
        else:
            self.mu = np.array(mu)
        
        if x0 is None:
            self.noise = np.array(self.mu)
        else:
            self.noise = np.array(x0*np.ones(action_dimensions))
        self.reset()
    
    def reset(self):
        # might need to implement this but make it so that self.noise is the same dimensions as action:
        # self.noise = self.mu if self.x0 is None else self.x0
        return 0
    
    def generate_noise(self):
        for dimension in range(action_dimensions):
            self.noise[dimension] =self.noise[dimension] + self.theta * (self.mu[dimension] - self.noise[dimension]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=1)
        return T.from_numpy(self.noise).float()

class critic_network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # State processing stream
        self.state_layer = nn.Linear(state_dim, 400)
        # Combined processing
        self.layer1 = nn.Linear(400 + action_dim, 300)
        self.layer2 = nn.Linear(300, 1)  # Outputs a single Q-value

    def forward(self, state, action):
        # Process state first, then combine with action
        state_value = F.relu(self.state_layer(state))
        combined = T.cat([state_value, action], dim=-1)
        x = F.relu(self.layer1(combined))
        q_value = self.layer2(x)
        return q_value
        