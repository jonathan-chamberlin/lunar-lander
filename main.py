import os 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2,dt=1e-2,x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

# create method within OUAActionNoise

class Customer(object):
    def __init__(self, name, email, phone):
        self.name = name
        self.email = email
        self.phone = phone
        self.reset()
    def send_sms(self):
        print(f"Send a sms to {self.name}'s number of {self.phone}")

customer1 = Customer("Jonathan", "jcham17x@gmail.com", 5426783357)
customer1.send_sms()