


import warnings
import os 
import pygame as pg
import gymnasium as gym
import random
import collections
from network import *
warnings.filterwarnings("ignore")
pg.init()



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


lunar_actor = actor_network(8,2)
lunar_critic = critic_network(8,2)

actor_optimizer = optim.Adam(lunar_actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(lunar_critic.parameters(), lr=0.001)

lunar_noise = OUActionNoise(mu,sigma,theta,dt,x0,action_dimensions)

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
                print(f"successes_list: {successes_list}")
                running = False
        
        
        
        noise = lunar_noise.generate_noise()
        action_without_noise = lunar_actor(state)
        # print(f"noise: {noise}")
        noisy_action = (action_without_noise + noise).float()
        action_calculations = training_env.step(noisy_action.detach().numpy())

        
        # print(f"Action: {action}")
        
        next_state = T.from_numpy(action_calculations[0])
        reward = action_calculations[1]
        terminated = action_calculations[2]
        truncated = action_calculations[3]
        info = action_calculations[4]
        
        q_value_for_actor = lunar_critic(state,noisy_action)

        actor_loss = -q_value_for_actor
        # print(f"actor_loss: {actor_loss[0]}")
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        q_value_for_critic = lunar_critic(state,noisy_action.detach())
        with T.no_grad():
            next_action = lunar_actor(next_state)
            target = reward + gamma * lunar_critic(next_state,next_action) #bellman
        critic_loss = F.mse_loss(q_value_for_critic,target.detach())
        # print(f"Critic Loss: {critic_loss[0]}")
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # print(f"State: {state}")
        experience = (T.tensor(state.detach()), T.tensor(noisy_action.detach()),T.tensor(reward),T.tensor(next_state.detach()), T.tensor(True))
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
    
    print(f"State batch: {state_batch}")
    print(f"reward_batch: {reward_batch}")
    print(f"next_state_batch: {next_state_batch}")
    print(f"done_flag_batch: {done_flag_batch}")
        
    print(f"total_reward_for_one_run: {total_reward_for_one_run}")
    total_reward_for_alls_runs.append(total_reward_for_one_run)



