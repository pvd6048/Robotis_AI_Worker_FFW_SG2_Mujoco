import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

env_id = "Ant-v5"
n_envs = 4  # parallel envs for faster data collection

train_env = make_vec_env(env_id, n_envs=n_envs)

eval_env = make_vec_env(env_id, n_envs=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/ant_ppo_best",
    log_path="./logs/ant_ppo",
    eval_freq=25_000,
    n_eval_episodes=5,
    deterministic=True,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    learning_rate=3e-4,
    tensorboard_log="./logs/tensorboard",
    verbose=1,
)

print(f"Training PPO on {env_id} with {n_envs} parallel envs")
print(f"Policy network: {model.policy}")
print("-" * 60)

model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)

model.save("models/ant_ppo_final")
print("Saved model to models/ant_ppo_final.zip")
