import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, TransformObservation
from gymnasium.spaces import Box
from dreamer import Dreamer
import numpy as np

env = gym.make("CarRacing-v3", continuous=True, render_mode='rgb_array')
env = ResizeObservation(env, (96, 96))
old_space = env.observation_space
new_space = Box(low=old_space.low.min(), high=old_space.high.max(), shape=(3, 96, 96), dtype=old_space.dtype)
env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)), new_space)
agent = Dreamer(env)
agent.run(env, run_id='dreamer_carracing')