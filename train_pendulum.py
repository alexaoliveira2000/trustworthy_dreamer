import gymnasium as gym
from dreamer import Dreamer

env = gym.make("Pendulum-v1", render_mode="rgb_array")
agent = Dreamer(env)
agent.run(env, run_id='dreamer_pendulum')