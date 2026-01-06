


import warnings
import os
import time
import pygame as pg
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import random
import collections
import numpy as np
from network import *
warnings.filterwarnings("ignore")
pg.init()

# Start timer if timing is enabled
if timing:
    start_time = time.time()



# Setup for selective rendering - only render runs in runs_to_render
runs_to_render_set = set(runs_to_render)  # O(1) lookup

# Vectorized environment for fast parallel training (no rendering)
vec_env = SyncVectorEnv([
    lambda: gym.make("LunarLanderContinuous-v3")
    for _ in range(num_envs)
])

# Single environment for rendered episodes
render_env = gym.make("LunarLanderContinuous-v3", render_mode="human")
render_env.metadata["render_fps"] = framerate

print(f"Vectorized env with {num_envs} parallel environments")

# print(f"Initial Observation: {initial_observation}")

'''
print("Action space:", env.action_space)
print("Action space shape:", env.action_space.shape)
print("Action space low:", env.action_space.low)
print("Action space high:", env.action_space.high)
print("Sample random action:", env.action_space.sample())
'''

lunar_actor = actor_network(8,2)
lunar_critic = critic_network(8,2)
lunar_critic_2 = critic_network(8,2)  # TD3: Second critic to reduce overestimation

# Target networks for stable training
target_actor = actor_network(8,2)
target_critic = critic_network(8,2)
target_critic_2 = critic_network(8,2)  # TD3: Second target critic

# Initialize target networks with same weights as main networks
target_actor.load_state_dict(lunar_actor.state_dict())
target_critic.load_state_dict(lunar_critic.state_dict())
target_critic_2.load_state_dict(lunar_critic_2.state_dict())

actor_optimizer = optim.Adam(lunar_actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(lunar_critic.parameters(), lr=critic_lr)
critic_optimizer_2 = optim.Adam(lunar_critic_2.parameters(), lr=critic_lr)  # TD3: Second critic optimizer

lunar_noise = OUActionNoise(mu, sigma, theta, dt, x0, action_dimensions, num_envs)

# Soft update function for target networks (polyak averaging)
def soft_update_target_networks(main_network, target_network, tau):
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# Reward shaping function to provide intermediate learning signals
def shape_reward(state, base_reward, done):
    """
    Minimal reward shaping: encourage descent, discourage hovering.
    LunarLander state: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]
    """
    shaped_reward = base_reward

    y_pos = state[1]
    y_vel = state[3]

    # Only reward active descent (not hovering)
    if (y_vel < -0.1):
        shaped_reward += 0.2
    
    if abs(y_vel) < 0.02:
        shaped_reward += 0.2

    # Small penalty for being high up (discourages hovering)
    if y_pos > 0.5:
        shaped_reward -= 1

    return shaped_reward

total_reward_for_alls_runs = []
successes_list =[]

# memory. Format is (state, action, reward, next_state, done_flag)
# Using a list with manual index management for O(1) random access
# (deque + random.sample is O(n) which kills performance)
experiences = []
experience_index = 0

def add_experience(exp):
    """Add experience to circular buffer with O(1) complexity."""
    global experience_index
    if len(experiences) < max_experiences_to_store:
        experiences.append(exp)
    else:
        experiences[experience_index] = exp
    experience_index = (experience_index + 1) % max_experiences_to_store

# Training step counter
total_training_steps = 0

# Flag to track if user wants to quit
user_quit = False

# Flag to track if warmup phase (initial random exploration) is complete
warmup_completed = False

# ===== DIAGNOSTIC TRACKING =====
# Track metrics over episodes for analysis
episode_q_values = []  # Average Q-value per episode
episode_actor_losses = []  # Actor loss per episode
episode_critic_losses = []  # Critic loss per episode
episode_action_means = []  # Mean action magnitude per episode
episode_action_stds = []  # Action std per episode
episode_main_thruster = []  # Average main thruster (action[0]) per episode
episode_side_thruster = []  # Average side thruster (action[1]) per episode
episode_actor_grad_norms = []  # Actor gradient norm per episode
episode_critic_grad_norms = []  # Critic gradient norm per episode


# ===== HELPER FUNCTION FOR TRAINING =====
def do_training_step():
    """Perform one training update on a random batch from experiences."""
    global total_training_steps

    random_sample_experiences = random.sample(experiences, sample_size if len(experiences) > sample_size else len(experiences))

    # format is (state, action, reward, next_state, done_flag)
    state_batch, action_batch, reward_batch, next_state_batch, done_flag_batch = zip(*random_sample_experiences)

    state_batch = T.stack(state_batch)
    action_batch = T.stack(action_batch)
    reward_batch = T.stack(reward_batch)
    next_state_batch = T.stack(next_state_batch)
    done_flag_batch = T.stack(done_flag_batch)

    # === CRITIC TRAINING (TD3: Use both critics) ===
    with T.no_grad():
        next_actions = target_actor(next_state_batch)
        noise = T.randn_like(next_actions) * target_policy_noise
        noise = T.clamp(noise, -target_noise_clip, target_noise_clip)
        next_actions_noisy = next_actions + noise
        next_actions_noisy[:, 0] = T.clamp(next_actions_noisy[:, 0], -1.0, 1.0)
        next_actions_noisy[:, 1] = T.clamp(next_actions_noisy[:, 1], -1.0, 1.0)

        target_q_values_1 = target_critic(next_state_batch, next_actions_noisy)
        target_q_values_2 = target_critic_2(next_state_batch, next_actions_noisy)
        target_q_values = T.min(target_q_values_1, target_q_values_2)

        done_mask = done_flag_batch.unsqueeze(-1).float()
        target_q = reward_batch.unsqueeze(-1) + gamma * target_q_values * (1.0 - done_mask)

    current_q_values_1 = lunar_critic(state_batch, action_batch)
    current_q_values_2 = lunar_critic_2(state_batch, action_batch)

    critic_loss_1 = F.mse_loss(current_q_values_1, target_q)
    critic_loss_2 = F.mse_loss(current_q_values_2, target_q)

    critic_optimizer.zero_grad()
    critic_loss_1.backward()
    critic_grad_norm_1 = T.nn.utils.clip_grad_norm_(lunar_critic.parameters(), gradient_clip_value)
    critic_optimizer.step()

    critic_optimizer_2.zero_grad()
    critic_loss_2.backward()
    critic_grad_norm_2 = T.nn.utils.clip_grad_norm_(lunar_critic_2.parameters(), gradient_clip_value)
    critic_optimizer_2.step()

    critic_loss = (critic_loss_1.item() + critic_loss_2.item()) / 2.0
    critic_grad = (critic_grad_norm_1.item() + critic_grad_norm_2.item()) / 2.0

    actor_loss = 0.0
    actor_grad = 0.0

    # === DELAYED ACTOR TRAINING (TD3) ===
    if total_training_steps % policy_update_frequency == 0:
        predicted_actions = lunar_actor(state_batch)
        actor_loss_tensor = -lunar_critic(state_batch, predicted_actions).mean()

        actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        actor_grad = T.nn.utils.clip_grad_norm_(lunar_actor.parameters(), gradient_clip_value).item()
        actor_optimizer.step()

        soft_update_target_networks(lunar_actor, target_actor, tau)
        soft_update_target_networks(lunar_critic, target_critic, tau)
        soft_update_target_networks(lunar_critic_2, target_critic_2, tau)

        actor_loss = actor_loss_tensor.item()

    total_training_steps += 1

    avg_q = ((current_q_values_1.mean() + current_q_values_2.mean()) / 2.0).item()
    return critic_loss, actor_loss, critic_grad, actor_grad, avg_q


# ===== HELPER FUNCTION FOR RENDERED EPISODE =====
def run_rendered_episode(episode_num):
    """Run a single episode on the render_env and return total reward."""
    global user_quit

    obs, _ = render_env.reset()
    state = T.from_numpy(obs).float()

    noise_scale = max(noise_scale_final,
                     noise_scale_initial - (noise_scale_initial - noise_scale_final) * episode_num / noise_decay_episodes)

    reward_list = []
    actions_list = []
    shaped_bonus = 0.0
    running = True

    print(f"Run {episode_num} (RENDERED)")

    while running:
        events = pg.event.get()
        for event in events:
            if event.type == 256:
                render_env.close()
                print("=" * 50)
                print(f"total_reward_for_alls_runs: {total_reward_for_alls_runs}")
                print(f"successes_list: {successes_list}")
                running = False
                user_quit = True
                break

        if not running:
            break

        # Generate action
        if episode_num < random_warmup_episodes:
            action = T.from_numpy(render_env.action_space.sample()).float()
        else:
            noise = lunar_noise.generate_noise_single() * noise_scale
            action = (lunar_actor(state) + noise).float()
            action[0] = T.clamp(action[0], -1.0, 1.0)
            action[1] = T.clamp(action[1], -1.0, 1.0)

        actions_list.append(action.detach().cpu().numpy())

        next_obs, reward, terminated, truncated, info = render_env.step(action.detach().numpy())
        next_state = T.from_numpy(next_obs).float()

        shaped_reward = shape_reward(obs, reward, terminated)
        shaped_bonus += (shaped_reward - reward)

        experience = (state.detach().clone(), action.detach().clone(),
                     T.tensor(shaped_reward, dtype=T.float32), next_state.detach().clone(),
                     T.tensor(terminated))
        add_experience(experience)

        state = next_state
        obs = next_obs
        reward_list.append(float(reward))

        if terminated or truncated:
            running = False

    total_reward = float(np.sum(reward_list))

    # Track action statistics
    if len(actions_list) > 0 and len(experiences) >= min_experiences_before_training:
        actions_array = np.array(actions_list)
        episode_action_means.append(np.mean(np.abs(actions_array)))
        episode_action_stds.append(np.std(actions_array))
        episode_main_thruster.append(np.mean(actions_array[:, 0]))
        episode_side_thruster.append(np.mean(actions_array[:, 1]))

    return total_reward, shaped_bonus


# ===== MAIN GAME LOOP (VECTORIZED) =====

# Initialize vectorized environment states
observations, _ = vec_env.reset()
states = T.from_numpy(observations).float()

# Per-environment tracking
env_rewards = [[] for _ in range(num_envs)]
env_shaped_bonus = [0.0 for _ in range(num_envs)]
env_actions = [[] for _ in range(num_envs)]

completed_episodes = 0
steps_since_training = 0
training_started = False

print(f"Starting training with {num_envs} parallel environments...")

while completed_episodes < runs and not user_quit:
    # Check if current episode should be rendered
    if completed_episodes in runs_to_render_set:
        total_reward, shaped_bonus = run_rendered_episode(completed_episodes)

        if not user_quit:
            if total_reward >= 200:
                successes_list.append(completed_episodes)
                print("SUCCESS")
            else:
                print("FAILURE")

            print(f"total_reward_for_one_run: {total_reward} (shaped bonus: +{shaped_bonus:.1f})")
            total_reward_for_alls_runs.append(total_reward)
            completed_episodes += 1
        continue

    # Calculate noise scale based on completed episodes
    noise_scale = max(noise_scale_final,
                     noise_scale_initial - (noise_scale_initial - noise_scale_final) * completed_episodes / noise_decay_episodes)

    # Generate actions for all environments
    if completed_episodes < random_warmup_episodes:
        # Random actions for all envs
        actions = T.from_numpy(vec_env.action_space.sample()).float()
    else:
        # Actor + noise for all envs
        with T.no_grad():
            noise = lunar_noise.generate_noise() * noise_scale
            actions = lunar_actor(states) + noise
            actions[:, 0] = T.clamp(actions[:, 0], -1.0, 1.0)
            actions[:, 1] = T.clamp(actions[:, 1], -1.0, 1.0)

    # Step all environments
    next_observations, rewards, terminateds, truncateds, infos = vec_env.step(actions.detach().numpy())
    next_states = T.from_numpy(next_observations).float()

    # Process each environment
    for i in range(num_envs):
        # Track actions
        env_actions[i].append(actions[i].detach().cpu().numpy())

        # Compute shaped reward
        shaped_reward = shape_reward(observations[i], rewards[i], terminateds[i])
        env_shaped_bonus[i] += (shaped_reward - rewards[i])

        # Store experience
        experience = (states[i].detach().clone(), actions[i].detach().clone(),
                     T.tensor(shaped_reward, dtype=T.float32), next_states[i].detach().clone(),
                     T.tensor(terminateds[i]))
        add_experience(experience)

        # Track original reward
        env_rewards[i].append(float(rewards[i]))

        # Check if episode completed (terminated OR truncated)
        if terminateds[i] or truncateds[i]:
            total_reward = float(np.sum(env_rewards[i]))

            print(f"Run {completed_episodes}")
            if total_reward >= 200:
                successes_list.append(completed_episodes)
                print("SUCCESS")
            else:
                print("FAILURE")

            # Track action statistics
            if len(env_actions[i]) > 0 and len(experiences) >= min_experiences_before_training:
                actions_array = np.array(env_actions[i])
                episode_action_means.append(np.mean(np.abs(actions_array)))
                episode_action_stds.append(np.std(actions_array))
                episode_main_thruster.append(np.mean(actions_array[:, 0]))
                episode_side_thruster.append(np.mean(actions_array[:, 1]))

            print(f"total_reward_for_one_run: {total_reward} (shaped bonus: +{env_shaped_bonus[i]:.1f})")
            total_reward_for_alls_runs.append(total_reward)

            # Reset tracking for this env
            env_rewards[i] = []
            env_shaped_bonus[i] = 0.0
            env_actions[i] = []
            lunar_noise.reset(i)

            completed_episodes += 1

            # Stop if we've reached target runs
            if completed_episodes >= runs:
                break

    # Update states
    states = next_states
    observations = next_observations
    steps_since_training += num_envs

    # Training: run updates based on step count
    if len(experiences) >= min_experiences_before_training:
        if not training_started:
            print(f">>> TRAINING STARTED at episode {completed_episodes} with {len(experiences)} experiences <<<")
            training_started = True

        # Train proportionally to steps taken (roughly training_updates_per_episode per ~200 steps)
        updates_to_do = max(1, steps_since_training // 4)  # ~50 updates per 200 steps

        total_critic_loss = 0
        total_actor_loss = 0
        total_critic_grad = 0
        total_actor_grad = 0
        total_q = 0
        actor_update_count = 0

        for _ in range(updates_to_do):
            c_loss, a_loss, c_grad, a_grad, avg_q = do_training_step()
            total_critic_loss += c_loss
            total_actor_loss += a_loss
            total_critic_grad += c_grad
            total_actor_grad += a_grad
            total_q += avg_q
            if a_loss != 0:
                actor_update_count += 1

        steps_since_training = 0

        # Log training metrics periodically
        if completed_episodes % 10 == 0 and updates_to_do > 0:
            avg_c_loss = total_critic_loss / updates_to_do
            avg_a_loss = total_actor_loss / max(actor_update_count, 1)
            avg_q_val = total_q / updates_to_do

            episode_q_values.append(avg_q_val)
            episode_actor_losses.append(avg_a_loss)
            episode_critic_losses.append(avg_c_loss)
            episode_actor_grad_norms.append(total_actor_grad / max(actor_update_count, 1))
            episode_critic_grad_norms.append(total_critic_grad / updates_to_do)

            print(f"Training update - Critic Loss: {avg_c_loss:.4f}, Actor Loss: {avg_a_loss:.4f}, Avg Q: {avg_q_val:.3f}, Noise: {noise_scale:.3f}")

# Close environments
vec_env.close()
render_env.close()


# ===== COMPREHENSIVE DIAGNOSTIC OUTPUT =====
print("\n--- DIAGNOSTIC CODE REACHED - PROCESSING RESULTS ---")
import sys
sys.stdout.flush()

print("\n" + "="*80)
print("TRAINING DIAGNOSTICS SUMMARY")
print("="*80)

# Reward statistics
print(f"\n--- REWARD STATISTICS ---")
print(f"Total episodes: {len(total_reward_for_alls_runs)}")
print(f"Successes: {len(successes_list)} (episodes: {successes_list})")

print(f"Success rate: {len(successes_list)/len(total_reward_for_alls_runs)*100:.1f}%")
print(f"Mean reward: {np.mean(total_reward_for_alls_runs):.2f}")
print(f"Max reward: {np.max(total_reward_for_alls_runs):.2f}")
print(f"Min reward: {np.min(total_reward_for_alls_runs):.2f}")
if len(total_reward_for_alls_runs) >= 50:
    print(f"Final 50 episodes mean reward: {np.mean(total_reward_for_alls_runs[-50:]):.2f}")
# Action statistics


print(f"\n--- ACTION STATISTICS ---")
if len(episode_main_thruster) > 0:
    print(f"Episodes with training: {len(episode_main_thruster)}")
    print(f"Mean main thruster (all episodes): {np.mean(episode_main_thruster):.3f}")
    print(f"Mean side thruster (all episodes): {np.mean(episode_side_thruster):.3f}")
    print(f"Mean action magnitude: {np.mean(episode_action_means):.3f}")
    print(f"Mean action std: {np.mean(episode_action_stds):.3f}")

    print(f"\nLast 50 episodes:")
    print(f"  Main thruster: {np.mean(episode_main_thruster[-50:]):.3f}")
    print(f"  Side thruster: {np.mean(episode_side_thruster[-50:]):.3f}")
    print(f"  Action magnitude: {np.mean(episode_action_means[-50:]):.3f}")

    # Check for blasting upward pattern
    high_thruster_episodes = sum(1 for x in episode_main_thruster if x > 0.5)
    print(f"\nEpisodes with high main thruster (>0.5): {high_thruster_episodes}/{len(episode_main_thruster)}")
else:
    print("No action data collected (training hasn't started yet)")


# Q-value and loss statistics

print(f"\n--- TRAINING METRICS ---")
if len(episode_q_values) > 0:
    print(f"Mean Q-value: {np.mean(episode_q_values):.3f}")
    if len(episode_q_values) >= 10:
        print(f"Q-value trend (first 10 vs last 10): {np.mean(episode_q_values[:10]):.3f} -> {np.mean(episode_q_values[-10:]):.3f}")
    print(f"Mean actor loss: {np.mean(episode_actor_losses):.4f}")
    print(f"Mean critic loss: {np.mean(episode_critic_losses):.4f}")
    print(f"Mean actor gradient norm: {np.mean(episode_actor_grad_norms):.4f}")
    print(f"Mean critic gradient norm: {np.mean(episode_critic_grad_norms):.4f}")

    # Check for divergence patterns
    high_actor_loss_episodes = sum(1 for x in episode_actor_losses if x > 1.0)
    print(f"\nEpisodes with high actor loss (>1.0): {high_actor_loss_episodes}/{len(episode_actor_losses)}")
else:
    print("No training metrics collected yet")


# Sample recent episode details
print(f"\n--- LAST 10 EPISODES DETAIL ---")
length = len(total_reward_for_alls_runs)
if length > 5:
    start_idx = len(total_reward_for_alls_runs) - 5
else:
    start_idx = length-1

for i in range(start_idx, length-1):
    reward = total_reward_for_alls_runs[i]
    episode_num = i
    status = "SUCCESS" if i in successes_list else "FAILURE"

    info_str = f"Ep {episode_num}: {status}, Reward: {reward:.1f}"

    # Calculate the correct index in the tracking lists
    # (tracking starts later than episode 0 since training starts later)
    tracking_idx = i - (len(total_reward_for_alls_runs) - len(episode_main_thruster))
    if 0 <= tracking_idx < len(episode_main_thruster):
        main = episode_main_thruster[tracking_idx]
        side = episode_side_thruster[tracking_idx]
        q_val = episode_q_values[tracking_idx] if tracking_idx < len(episode_q_values) else 0
        info_str += f", Main: {main:.2f}, Side: {side:.2f}, Q: {q_val:.2f}"

    print(info_str)

# Key data section - printed once, outside the loop
print("\n" + "="*80)
print("KEY DATA FOR ANALYSIS")
print("="*80)
print(f"\nReward list (last 50): {total_reward_for_alls_runs[-50:]}")
print(f"\nMain thruster list (last 50): {episode_main_thruster[-50:] if len(episode_main_thruster) >= 50 else episode_main_thruster}")
print(f"\nQ-values list (last 50): {episode_q_values[-50:] if len(episode_q_values) >= 50 else episode_q_values}")
print(f"\nActor losses list (last 50): {episode_actor_losses[-50:] if len(episode_actor_losses) >= 50 else episode_actor_losses}")

print("\n" + "="*80)
print("END OF DIAGNOSTICS")
print("="*80)

# Print elapsed time if timing is enabled
if timing:
    elapsed_time = time.time() - start_time
    print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")
