import torch
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

gym.register_envs(gymnasium_robotics)

env_id = "FetchReach-v4"

train_env = gym.make(env_id)
eval_env  = gym.make(env_id)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/fetch_her_sac_best",
    log_path="./logs/fetch_her_sac",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

model = SAC(
    "MultiInputPolicy",       # goal-conditioned obs dict needs this
    train_env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,     # HER: sample 4 virtual goals per real transition
        goal_selection_strategy="future",
    ),
    learning_rate=1e-3,
    buffer_size=1_000_000,
    batch_size=512,
    gamma=0.95,
    tau=0.05,
    learning_starts=1000,
    device=device,
    tensorboard_log="./logs/tensorboard",
    verbose=1,
)

print(f"Training SAC + HER on {env_id}")
print(f"Obs space   : {train_env.observation_space}")
print(f"Action space: {train_env.action_space}")
print("-" * 60)

model.learn(total_timesteps=200_000, callback=eval_callback, progress_bar=True)

model.save("models/fetch_her_sac_final")
print("Saved model to models/fetch_her_sac_final.zip")

train_env.close()
eval_env.close()
