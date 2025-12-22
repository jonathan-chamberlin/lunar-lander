import gymnasium as gym
pg.init()
# Try this and observe what happens

env = gym.make("LunarLander-v3", render_mode="human")
env.reset()
running = True
while running:    
    events = pg.event.get()
    for event in events:
        if event.type == 256:
            running = False
            env.close()
    # print(f"Event type: {event.type}")
    # print(f"Event object: {event}")
    
    
    
    
