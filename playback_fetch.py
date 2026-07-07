import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import time

gym.register_envs(gymnasium_robotics)

model = SAC.load("models/fetch_her_sac_best/best_model")

env = gym.make("FetchReach-v4", render_mode="human")

n_episodes = 10
successes = 0

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.02)
    success = info.get("is_success", False)
    successes += int(success)
    print(f"Episode {ep + 1:2d}: {'SUCCESS' if success else 'FAIL'}")

print(f"\nSuccess rate: {successes}/{n_episodes} ({100 * successes / n_episodes:.0f}%)")
env.close()
