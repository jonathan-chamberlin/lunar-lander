framerate = 600

runs = 500
runs_to_render = [0,3,4]
gamma = 0.99
alpha = 0.1

mu = [0,0]
sigma = 0.1
theta = 0.2
dt = 0.01
x0 = 0
action_dimensions = 2

max_experiences_to_store = 1<<16

sample_size = 1<<7
tau = 0.005  # Soft update rate for target networks
min_experiences_before_training = 1000  # Minimum buffer size before training starts