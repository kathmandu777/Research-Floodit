from gym.envs.registration import register

register(
    id='floodit-v0',
    entry_point='gym_floodit.envs:FlooditEnv'
)
