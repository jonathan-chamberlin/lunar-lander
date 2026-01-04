framerate = 600

runs = 500
runs_to_render = [0,3,4]
gamma = 0.99
alpha = 0.1
actor_lr = 0.00003  # Actor learning rate (very slow for stability)
critic_lr = 0.0003  # Critic learning rate (10x actor)

mu = [0,0]
sigma = 0.1
theta = 0.2
dt = 0.01
x0 = 0
action_dimensions = 2

max_experiences_to_store = 1<<16

sample_size = 1<<7
tau = 0.001  # Soft update rate for target networks (slow for stability)
min_experiences_before_training = 5000  # Minimum buffer size before training starts

# Noise decay parameters
noise_scale_initial = 1.0  # Start with full noise
noise_scale_final = 0.05   # End with 5% noise (more exploitation)
noise_decay_episodes = 300  # Decay over first 300 episodes (slow decay for stability)

# Training parameters
training_updates_per_episode = 4  # Number of gradient updates per episode (reduced for stability)
reward_scale = 0.01  # Scale rewards to prevent large Q-values
gradient_clip_value = 1.0  # Clip gradients to prevent divergence