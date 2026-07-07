import mujoco
import gymnasium as gym
import numpy as np
import time

print(f"MuJoCo {mujoco.__version__} | Gymnasium {gym.__version__}")
print("-" * 40)

env = gym.make("Ant-v5", render_mode="human")
obs, info = env.reset(seed=0)

print(f"Environment : Ant-v5")
print(f"Obs space   : {env.observation_space}")
print(f"Action space: {env.action_space}")
print()

total_reward = 0
steps = 200
for i in range(steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.02)
    if terminated or truncated:
        obs, info = env.reset()

env.close()

print(f"Ran {steps} steps with random actions")
print(f"Total reward: {total_reward:.2f}")
print("MuJoCo is working correctly.")
