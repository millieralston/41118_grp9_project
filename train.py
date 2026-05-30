import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
# import husky_env_neo   # Change to your filename (without .py)
import husky_chaser_env as husky_env_neo   # CNN image-based env

class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0 and self.num_timesteps > 0:
            print(f"[Training Progress] timesteps={self.num_timesteps}", flush=True)
        return True

    def _on_rollout_start(self) -> None:
        print(f"[Rollout Start] timesteps={self.num_timesteps}", flush=True)

    def _on_rollout_end(self) -> None:
        print(f"[Rollout Completed] timesteps={self.num_timesteps}", flush=True)

# Register environment
def make_env():
    # env = husky_env_neo.HuskyChaserEnv()
    env = husky_env_neo.HuskyChaserEnv(render_mode="direct")   # Use DIRECT for training
    return env

# Vectorized environments (parallel training)
vec_env = make_vec_env(make_env, n_envs=4)
vec_env = VecTransposeImage(vec_env)      # Important for CNN
vec_env = VecMonitor(vec_env)

model = PPO(
    "CnnPolicy", 
    vec_env,
    verbose=1,
    tensorboard_log="./husky_ppo_logs/",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,           # Exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="auto"
)

# Save best model + checkpoints
eval_callback = EvalCallback(vec_env, best_model_save_path="./best_model/",
                             log_path="./logs/", eval_freq=5000,
                             n_eval_episodes=5, deterministic=True)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./checkpoints/")
progress_callback = ProgressCallback(print_freq=1000)

print("Starting PPO Training...", flush=True)
model.learn(total_timesteps=200_000, callback=[progress_callback, eval_callback, checkpoint_callback])

model.save("husky_chaser_ppo_final")
print("Training finished!", flush=True)