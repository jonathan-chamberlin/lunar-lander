timing = True  # Set to True to benchmark simulation time
framerate = 600
runs = 400
num_envs = 8  # Number of parallel environments for vectorized training

all = list(range(runs-1))
last = runs-1

# runs_to_render = [100,101,102,103,104,200,201,202,203, 300,301,302,303,396,397,398,399]
runs_to_render = all
gamma = 0.99
alpha = 0.1
actor_lr = 0.0001  # Actor learning rate (reduced to prevent action saturation)
critic_lr = 0.0003  # Critic learning rate

mu = [0,0]
sigma = 0.1
theta = 0.2
dt = 0.01
x0 = 0
action_dimensions = 2

max_experiences_to_store = 1<<16

sample_size = 1<<8  # 256 batch size (TD3 standard - 2x increase for stability)
tau = 0.001  # Soft update rate for target networks (reduced for stability)
min_experiences_before_training = 500  # Minimum buffer size before training starts (TD3 standard - 4x reduction to avoid poisoning)

# Noise decay parameters
noise_scale_initial = 1.0  # Start with full noise
noise_scale_final = 0.2   # End with 20% noise (higher floor to prevent policy collapse)
noise_decay_episodes = 300  # Decay over first 300 episodes (slower decay for more exploration)

# Random warmup: use completely random actions for first N episodes
random_warmup_episodes = 5

# Training parameters
training_updates_per_episode = 50  # Number of gradient updates per episode (TD3 standard - 10x increase)
reward_scale = 1.0  # NO SCALING - use raw rewards (now clipped to [-300, 300])
gradient_clip_value = 1.0  # Clip gradients

# TD3-style improvements for stability
policy_update_frequency = 2  # Update actor every N critic updates (TD3 standard)
target_policy_noise = 0.1  # Noise added to target actions for smoothing (reduced from 0.2)
target_noise_clip = 0.3  # Clip target noise to this range (reduced from 0.5)