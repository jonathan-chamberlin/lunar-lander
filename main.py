


import warnings
import os
import pygame as pg
import gymnasium as gym
import random
import collections
import numpy as np
from network import *
warnings.filterwarnings("ignore")
pg.init()



training_env = gym.make("LunarLanderContinuous-v3", render_mode="human")
render_env = gym.make("LunarLanderContinuous-v3", render_mode=None)
initial_observation = training_env.reset()[0]
state = T.from_numpy(training_env.reset()[0]).float()

training_env.metadata["render_fps"] = framerate
render_env.metadata["render_fps"] = framerate

print(f"NAME OF TEST: {training_env}")

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

lunar_noise = OUActionNoise(mu,sigma,theta,dt,x0,action_dimensions)

# Soft update function for target networks (polyak averaging)
def soft_update_target_networks(main_network, target_network, tau):
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# Reward shaping function to provide intermediate learning signals
def shape_reward(state, base_reward, done):
    """
    Add intermediate rewards to guide learning before achieving successful landings.

    LunarLander state: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]
    """
    shaped_reward = base_reward

    # Extract state components
    x_pos = state[0]
    y_pos = state[1]  # Altitude (height above ground)
    x_vel = state[2]
    y_vel = state[3]
    angle = state[4]
    angular_vel = state[5]
    leg1_contact = state[6]
    leg2_contact = state[7]

    # Reward 1: Staying upright (small constant reward)
    if abs(angle) < 0.3:  # Within ~17 degrees of upright
        shaped_reward += 0.2

    # Reward 2: Low velocity bonus (want to slow down for landing)
    total_velocity = np.sqrt(x_vel**2 + y_vel**2)
    if total_velocity < 1.0:
        shaped_reward += 0.5

    # Reward 3: Hovering near ground (preparing to land)
    if y_pos < 0.5 and total_velocity < 0.5 and abs(angle) < 0.2:
        shaped_reward += 1.0  # Reward for controlled hover near landing zone

    # Reward 4: Slow descent (negative y_vel is good when near ground)
    if y_pos < 0.8 and -0.5 < y_vel < 0.0:  # Slow downward motion near ground
        shaped_reward += 0.5

    # SOLUTION 2: Altitude-scaled progressive rewards
    # Give exponentially higher rewards the closer to ground with low velocity
    # This creates a strong gradient toward controlled descent
    altitude_factor = max(0, 1 - y_pos)  # 0 at top, 1 at ground
    velocity_quality = 1.0 / (1.0 + total_velocity)  # High when slow, low when fast
    altitude_velocity_bonus = altitude_factor * velocity_quality * 10.0
    shaped_reward += altitude_velocity_bonus

    # SOLUTION 4: Landing quality cumulative bonus
    # Reward sustained good behavior: near ground + slow + upright
    # This accumulates over multiple steps when all conditions are maintained
    if y_pos < 0.6 and total_velocity < 1.0 and abs(angle) < 0.4:
        upright_quality = 1.0 - abs(angle) / 0.4  # 1.0 when perfectly upright, 0.0 at threshold
        landing_quality_bonus = altitude_factor * velocity_quality * upright_quality * 8.0
        shaped_reward += landing_quality_bonus

    return shaped_reward

total_reward_for_alls_runs = []
successes_list =[]

# memory. Format is (state, action, reward, next_state, done_flag)
experiences = collections.deque(maxlen= max_experiences_to_store)

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


# Game loop

for run in range(0,runs):
    # Check if user closed the window in previous episode
    if user_quit:
        break

    #PSUEDOCODE for making it so only certain run indexes are rendered:
    # if we want to render the run:
        #if previous run wasn't rendered:
            #destroy training environment
        #use rendering environement
    # if we don't want to render the run
        #if previous run was rendered:
            #destroy rendering environment
        #use non-rendering environment
    
    train_obs, _ = training_env.reset()
    render_obs, _ = render_env.reset()
    state = T.from_numpy(train_obs).float()

    # Calculate noise scale for this episode (decay over episodes)
    noise_scale = max(noise_scale_final,
                     noise_scale_initial - (noise_scale_initial - noise_scale_final) * run / noise_decay_episodes)

    reward_list_for_run = []
    shaped_reward_bonus = 0.0  # Track total bonus from reward shaping
    actions_this_episode = []  # Track all actions in this episode
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
                print(f"successes_list: {successes_list}")
                running = False
                user_quit = True
                break  # Exit event loop

        # Skip rest of episode if user quit
        if not running:
            break

        noise = lunar_noise.generate_noise() * noise_scale
        action_without_noise = lunar_actor(state)

        # Actor now directly outputs correct ranges:
        # action[0] = main engine in [0, 1] (sigmoid)
        # action[1] = side engine in [-1, 1] (tanh)
        noisy_action = (action_without_noise + noise).float()

        # Clamp to valid ranges after adding noise
        noisy_action[0] = T.clamp(noisy_action[0], 0.0, 1.0)  # Main engine [0, 1]
        noisy_action[1] = T.clamp(noisy_action[1], -1.0, 1.0)  # Side engine [-1, 1]

        # Track actions for diagnostics
        actions_this_episode.append(noisy_action.detach().cpu().numpy())

        action_calculations = training_env.step(noisy_action.detach().numpy())


        # print(f"Action: {action}")

        next_state = T.from_numpy(action_calculations[0]).float()
        reward = action_calculations[1]
        terminated = action_calculations[2]
        truncated = action_calculations[3]
        info = action_calculations[4]

        # REWARD SHAPING: Add intermediate rewards to guide learning
        # IMPORTANT: Use train_obs (current state BEFORE step), not action_calculations[0] (next state AFTER step)
        shaped_reward = shape_reward(train_obs, reward, terminated)
        shaped_reward_bonus += (shaped_reward - reward)  # Track bonus added

        # Store experience with shaped reward (no clipping - let TD3 handle it)
        experience = (state.detach().clone(), noisy_action.detach().clone(), T.tensor(shaped_reward, dtype=T.float32), next_state.detach().clone(), T.tensor(terminated))
        experiences.append(experience)

        state = next_state
        train_obs = action_calculations[0]  # Update current observation for next iteration

        # Track ORIGINAL reward for success detection (not shaped)
        reward_list_for_run.append(float(reward))
        # print("=" * 10)
        # print(f"Original Reward: {reward:.2f}, Shaped Reward: {shaped_reward:.2f}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")

        if terminated == True:
            total_reward_for_one_run = float(np.sum(reward_list_for_run))
            if total_reward_for_one_run >= 200:
                successes_list.append(run)
                print("SUCCESS")
            else:
                print("FAILURE")
            training_env.reset()
            render_env.reset()
            running = False

    # Calculate total reward for this episode
    total_reward_for_one_run = float(np.sum(reward_list_for_run))

    # ===== COMPUTE ACTION STATISTICS FOR THIS EPISODE =====
    # Only track actions for episodes where training occurred
    if len(actions_this_episode) > 0 and len(experiences) >= min_experiences_before_training:
        actions_array = np.array(actions_this_episode)
        episode_action_means.append(np.mean(np.abs(actions_array)))
        episode_action_stds.append(np.std(actions_array))
        episode_main_thruster.append(np.mean(actions_array[:, 0]))  # Main engine (vertical)
        episode_side_thruster.append(np.mean(actions_array[:, 1]))  # Side engine (horizontal)

    # After each run ===================

    # Only train if we have enough experiences
    if len(experiences) >= min_experiences_before_training:
        if total_training_steps == 0:
            print(f">>> TRAINING STARTED at episode {run} with {len(experiences)} experiences <<<")

        total_critic_loss = 0
        total_actor_loss = 0
        actor_updates_count = 0
        total_actor_grad_norm = 0
        total_critic_grad_norm = 0

        # Multiple training updates per episode
        for update_step in range(training_updates_per_episode):
            random_sample_experiences = random.sample(experiences, sample_size if len(experiences) > sample_size else len(experiences))

            # format is  (state, action, reward, next_state, done_flag)
            state_batch, action_batch, reward_batch, next_state_batch, done_flag_batch = zip(*random_sample_experiences)

            state_batch = T.stack(state_batch)
            action_batch = T.stack(action_batch)
            reward_batch = T.stack(reward_batch)
            next_state_batch = T.stack(next_state_batch)
            done_flag_batch = T.stack(done_flag_batch)

            # === CRITIC TRAINING (TD3: Use both critics) ===
            # Compute target Q-values using target networks with policy smoothing (TD3)
            with T.no_grad():
                # Get actions for next states from target actor
                # Actor directly outputs: [0,1] for main, [-1,1] for side
                next_actions = target_actor(next_state_batch)

                # Add clipped noise to target actions for smoothing (prevents Q-overestimation)
                noise = T.randn_like(next_actions) * target_policy_noise
                noise = T.clamp(noise, -target_noise_clip, target_noise_clip)
                next_actions_noisy = next_actions + noise

                # Clamp to valid environment ranges
                next_actions_noisy[:, 0] = T.clamp(next_actions_noisy[:, 0], 0.0, 1.0)  # Main engine [0,1]
                next_actions_noisy[:, 1] = T.clamp(next_actions_noisy[:, 1], -1.0, 1.0)  # Side engine [-1,1]

                # TD3: Compute target Q-values from BOTH critics and take minimum
                target_q_values_1 = target_critic(next_state_batch, next_actions_noisy)
                target_q_values_2 = target_critic_2(next_state_batch, next_actions_noisy)
                target_q_values = T.min(target_q_values_1, target_q_values_2)

                # Apply Bellman equation with done mask (zero out future rewards for terminal states)
                done_mask = done_flag_batch.unsqueeze(-1).float()
                target_q = reward_batch.unsqueeze(-1) + gamma * target_q_values * (1.0 - done_mask)

            # Get current Q-values from BOTH critics
            current_q_values_1 = lunar_critic(state_batch, action_batch)
            current_q_values_2 = lunar_critic_2(state_batch, action_batch)

            # Compute critic losses (MSE between current Q and target Q) for both critics
            critic_loss_1 = F.mse_loss(current_q_values_1, target_q)
            critic_loss_2 = F.mse_loss(current_q_values_2, target_q)

            # Update critic 1
            critic_optimizer.zero_grad()
            critic_loss_1.backward()
            critic_grad_norm_1 = T.nn.utils.clip_grad_norm_(lunar_critic.parameters(), gradient_clip_value)
            critic_optimizer.step()

            # Update critic 2
            critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            critic_grad_norm_2 = T.nn.utils.clip_grad_norm_(lunar_critic_2.parameters(), gradient_clip_value)
            critic_optimizer_2.step()

            # Track combined critic metrics
            total_critic_loss += (critic_loss_1.item() + critic_loss_2.item()) / 2.0
            total_critic_grad_norm += (critic_grad_norm_1.item() + critic_grad_norm_2.item()) / 2.0

            # === DELAYED ACTOR TRAINING (TD3) ===
            # Only update actor every policy_update_frequency steps
            if update_step % policy_update_frequency == 0:
                # Compute actor loss: negative mean Q-value
                # Actor learns to maximize Q(s, actor(s))
                # Actor directly outputs correct action ranges
                predicted_actions = lunar_actor(state_batch)
                actor_loss = -lunar_critic(state_batch, predicted_actions).mean()

                # Update actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                # Capture gradient norm before clipping
                actor_grad_norm = T.nn.utils.clip_grad_norm_(lunar_actor.parameters(), gradient_clip_value)
                actor_optimizer.step()

                # === SOFT UPDATE TARGET NETWORKS (TD3: Update all three target networks) ===
                # Only update targets when actor updates
                soft_update_target_networks(lunar_actor, target_actor, tau)
                soft_update_target_networks(lunar_critic, target_critic, tau)
                soft_update_target_networks(lunar_critic_2, target_critic_2, tau)

                total_actor_loss += actor_loss.item()
                total_actor_grad_norm += actor_grad_norm.item()
                actor_updates_count += 1

            total_training_steps += 1

        avg_critic_loss = total_critic_loss / training_updates_per_episode
        avg_actor_loss = total_actor_loss / max(actor_updates_count, 1)  # Avoid division by zero
        avg_critic_grad = total_critic_grad_norm / training_updates_per_episode
        avg_actor_grad = total_actor_grad_norm / max(actor_updates_count, 1)

        # Compute average Q-value for diagnostics (using last batch, average both critics)
        with T.no_grad():
            avg_q_value = ((current_q_values_1.mean() + current_q_values_2.mean()) / 2.0).item()

        # Store metrics for analysis
        episode_q_values.append(avg_q_value)
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)
        episode_actor_grad_norms.append(avg_actor_grad)
        episode_critic_grad_norms.append(avg_critic_grad)

        print(f"Critic Loss: {avg_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}, Avg Q: {avg_q_value:.3f}, Noise: {noise_scale:.3f}")
        print(f"  Grads - Actor: {avg_actor_grad:.4f}, Critic: {avg_critic_grad:.4f}")
        print(f"  Actions - Main: {episode_main_thruster[-1]:.3f}, Side: {episode_side_thruster[-1]:.3f}, Mean: {episode_action_means[-1]:.3f}")

        # Warning if actor loss is too positive (indicates divergence)
        if avg_actor_loss > 5:
            print(f"  ⚠️  WARNING: High actor loss ({avg_actor_loss:.2f}) - possible divergence!")

        # Warning if main thruster is too high (blasting upward behavior)
        if episode_main_thruster[-1] > 0.5:
            print(f"  ⚠️  WARNING: High main thruster ({episode_main_thruster[-1]:.2f}) - blasting upward!")    
    
    print(f"total_reward_for_one_run: {total_reward_for_one_run} (shaped bonus: +{shaped_reward_bonus:.1f})")
    total_reward_for_alls_runs.append(total_reward_for_one_run)


# ===== COMPREHENSIVE DIAGNOSTIC OUTPUT =====
print("\n🔍 DIAGNOSTIC CODE REACHED - PROCESSING RESULTS...")
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
print(f"Episodes with training: {len(episode_main_thruster)}")
print(f"Mean main thruster (all episodes): {np.mean(episode_main_thruster):.3f}")
print(f"Mean side thruster (all episodes): {np.mean(episode_side_thruster):.3f}")
print(f"Mean action magnitude: {np.mean(episode_action_means):.3f}")
print(f"Mean action std: {np.mean(episode_action_stds):.3f}")

    # Last 50 episodes

print(f"\nLast 50 episodes:")
print(f"  Main thruster: {np.mean(episode_main_thruster[-50:]):.3f}")
print(f"  Side thruster: {np.mean(episode_side_thruster[-50:]):.3f}")
print(f"  Action magnitude: {np.mean(episode_action_means[-50:]):.3f}")

# Check for blasting upward pattern
high_thruster_episodes = sum(1 for x in episode_main_thruster if x > 0.5)
print(f"\nEpisodes with high main thruster (>0.5): {high_thruster_episodes}/{len(episode_main_thruster)}")

print(f"\n--- ACTION STATISTICS ---")
print("No action data collected (training hasn't started yet)")


# Q-value and loss statistics

print(f"\n--- TRAINING METRICS ---")
print(f"Mean Q-value: {np.mean(episode_q_values):.3f}")
if len(episode_q_values) >= 10:
    print(f"Q-value trend (first 10 vs last 10): {np.mean(episode_q_values[:10]):.3f} → {np.mean(episode_q_values[-10:]):.3f}")
print(f"Mean actor loss: {np.mean(episode_actor_losses):.4f}")
print(f"Mean critic loss: {np.mean(episode_critic_losses):.4f}")
print(f"Mean actor gradient norm: {np.mean(episode_actor_grad_norms):.4f}")
print(f"Mean critic gradient norm: {np.mean(episode_critic_grad_norms):.4f}")

# Check for divergence patterns
high_actor_loss_episodes = sum(1 for x in episode_actor_losses if x > 1.0)
print(f"\nEpisodes with high actor loss (>1.0): {high_actor_loss_episodes}/{len(episode_actor_losses)}")


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

