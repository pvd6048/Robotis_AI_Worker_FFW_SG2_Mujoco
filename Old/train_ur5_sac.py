import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


def make_ur5_env():
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,   # dense rewards — no HER needed
        control_freq=20,
    )
    return GymWrapper(env)


train_env = make_ur5_env()
eval_env  = make_ur5_env()

print(f"UR5e Lift task")
print(f"Obs space   : {train_env.observation_space}")
print(f"Action space: {train_env.action_space}")
print("-" * 60)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/ur5_sac_best",
    log_path="./logs/ur5_sac",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=512,
    gamma=0.99,
    tau=0.005,
    learning_starts=5000,
    device=device,
    tensorboard_log="./logs/tensorboard",
    verbose=1,
)

model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True)

model.save("models/ur5_sac_final")
print("Saved model to models/ur5_sac_final.zip")

train_env.close()
eval_env.close()
