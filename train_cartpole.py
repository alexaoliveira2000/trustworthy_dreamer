import gymnasium as gym
from dreamer import Dreamer

env = gym.make("CartPole-v1", render_mode="rgb_array")
agent = Dreamer(env, batch_length=10)
agent.run(env, replay_ratio=256, run_id='dreamer_cartpole')