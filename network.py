from inputs import *
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class actor_network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size,256)
        self.layer2 = nn.Linear(256,output_size)

    def forward(self, state: T.tensor):
        action = T.tanh(self.layer2(F.relu(self.layer1(state)))) #the F.relu and T.tanh is the hyperbolic tangent to make this network non-linear, and to bound the tensor to values -1 to 1.
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
        return T.from_numpy(self.noise)

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
        