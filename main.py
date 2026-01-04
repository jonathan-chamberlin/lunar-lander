


import warnings
import os 
import pygame as pg
import gymnasium as gym
import random
import collections
from network import *
from functions import *
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

# Target networks for stable training
target_actor = actor_network(8,2)
target_critic = critic_network(8,2)

# Initialize target networks with same weights as main networks
target_actor.load_state_dict(lunar_actor.state_dict())
target_critic.load_state_dict(lunar_critic.state_dict())

actor_optimizer = optim.Adam(lunar_actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(lunar_critic.parameters(), lr=0.001)

lunar_noise = OUActionNoise(mu,sigma,theta,dt,x0,action_dimensions)

# Soft update function for target networks (polyak averaging)
def soft_update_target_networks(main_network, target_network, tau):
    for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

total_reward_for_alls_runs = []
successes_list =[]

# memory. Format is (state, action, reward, next_state, done_flag)
experiences = collections.deque(maxlen= max_experiences_to_store)


# Game loop

for run in range(0,runs):
    
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
                print(f"successes_list: {successes_list}")
                running = False
        
        
        
        noise = lunar_noise.generate_noise()
        action_without_noise = lunar_actor(state)
        # print(f"noise: {noise}")
        noisy_action = (action_without_noise + noise).float()
        action_calculations = training_env.step(noisy_action.detach().numpy())


        # print(f"Action: {action}")

        next_state = T.from_numpy(action_calculations[0]).float()
        reward = action_calculations[1]
        terminated = action_calculations[2]
        truncated = action_calculations[3]
        info = action_calculations[4]
        
        # COPIED LOSS AND FORWARD PASS CODE FROM HERE
        
        # print(f"State: {state}")
        experience = (state.detach().clone(), noisy_action.detach().clone(), T.tensor(reward, dtype=T.float32), next_state.detach().clone(), T.tensor(terminated))
        experiences.append(experience)
        
        state = next_state

        reward_list_for_run.append(float(reward))
        # print("=" * 10)
        # print(f"Observation: {observation}")
        # print(f"Reward: {reward}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")
        
        total_reward_for_one_run = float(np.sum(reward_list_for_run))
        
        if terminated == True:
            if total_reward_for_one_run >= 200:
                successes_list.append(run)
                print("SUCCESS")
            else:
                print("FAILURE")
            training_env.reset()
            render_env.reset()
            running = False
        
        
        
        
    # After each run ===================
    
    random_sample_experiences = random.sample(experiences,sample_size if len(experiences) > sample_size else len(experiences))
    print(f"Length of random_sample_experiences: {len(random_sample_experiences)}")
    # print(f"random_sample_experiences: {random_sample_experiences}")
    
    # format is  (state, action, reward, next_state, done_flag)
    state_batch, action_batch, reward_batch, next_state_batch, done_flag_batch = zip(*random_sample_experiences)
    
    state_batch = T.stack(state_batch)
    action_batch = T.stack(action_batch)
    reward_batch = T.stack(reward_batch)
    next_state_batch = T.stack(next_state_batch)
    done_flag_batch = T.stack(done_flag_batch)
    
    # Only train if we have enough experiences
    if len(experiences) >= min_experiences_before_training:

        # === CRITIC TRAINING ===
        # Compute target Q-values using target networks
        with T.no_grad():
            # Get actions for next states from target actor
            next_actions = target_actor(next_state_batch)
            # Compute target Q-values: r + gamma * Q_target(s', a')
            target_q_values = target_critic(next_state_batch, next_actions)
            # Apply Bellman equation
            target_q = reward_batch.unsqueeze(-1) + gamma * target_q_values

        # Get current Q-values from critic
        current_q_values = lunar_critic(state_batch, action_batch)

        # Compute critic loss (MSE between current Q and target Q)
        critic_loss = F.mse_loss(current_q_values, target_q)

        # Update critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # === ACTOR TRAINING ===
        # Compute actor loss: negative mean Q-value
        # Actor learns to maximize Q(s, actor(s))
        predicted_actions = lunar_actor(state_batch)
        actor_loss = -lunar_critic(state_batch, predicted_actions).mean()

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # === SOFT UPDATE TARGET NETWORKS ===
        soft_update_target_networks(lunar_actor, target_actor, tau)
        soft_update_target_networks(lunar_critic, target_critic, tau)

        print(f"Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")    
    
    print(f"total_reward_for_one_run: {total_reward_for_one_run}")
    total_reward_for_alls_runs.append(total_reward_for_one_run)



