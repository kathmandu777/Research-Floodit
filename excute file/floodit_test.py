
import gym
import gym_floodit

env = gym.make('floodit-v0')

observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # random action
    observation, reward, done, info = env.step(action)

    if done:
        env.reset()
