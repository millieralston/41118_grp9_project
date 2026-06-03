import argparse
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter

from husky_chaser_env import HuskyChaserEnv

# Old CNN/image-based imports kept for reference.
# import gymnasium as gym
# from stable_baselines3.common.vec_env import VecTransposeImage
# from stable_baselines3.common.callbacks import EvalCallback
# import husky_env_neo   # Old filename from lier version; not in this repository.


# Longer training kept for reference.
# TOTAL_TIMESTEPS = 200_000

# PPO needs more than a smoke-test run to learn stable pursuit and obstacle
# avoidance. 120k timesteps is the default training budget used here; lower it
# only when checking that the pipeline starts.
PREVIEW_FRAME_PATH = "preview_frame.png"
TOTAL_TIMESTEPS = 120_000


class PreviewCallback(BaseCallback):
    """Saves a camera frame to disk every `freq` steps for the GUI to display."""
    def __init__(self, base_env, freq=256):
        super().__init__()
        self._base_env = base_env
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            try:
                rgb = self._base_env.get_preview_image()
                cv2.imwrite(PREVIEW_FRAME_PATH, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            except Exception:
                pass
        return True
TENSORBOARD_LOG_DIR = "./husky_ppo_logs/"
CHECKPOINT_DIR = "./checkpoints/"
FINAL_MODEL_PATH = "husky_chaser_ppo_final"
MODEL_DIR = Path("models")

# PPO already logs generic learning values, but the project needs task-specific
# evidence too. This callback records pursuit, capture, collision, wall, and
# stuck metrics from the environment info dict so TensorBoard can show whether
# reward shaping is improving the actual chase behavior.
class TaskMetricsCallback(BaseCallback):
    """Record environment-specific pursuit and avoidance signals in TensorBoard."""

    # Continuous rollout metrics show how behavior changes during training:
    # distance should trend down, obstacle/wall clearance should stay healthy,
    # and near_target should increase as the chaser learns to pressure the runner.
    CONTINUOUS_KEYS = (
        "distance_to_runner",
        "chaser_movement",
        "min_obstacle_distance",
        "wall_clearance",
        "near_target",
    )
    # Episode flags summarize final outcomes: caught/is_success is the goal,
    # while collision, fell_over, ended_stuck, and truncated describe failures.
    EPISODE_KEYS = (
        "collision",
        "caught",
        "is_success",
        "fell_over",
        "ended_stuck",
        "truncated",
    )

    # The callback lifecycle methods are called at different stages of the training process.
    # _on_training_start is called once at the beginning of training, where we initialize counters and the TensorBoard writer.
    def _on_training_start(self) -> None:
        self._episode_count = 0
        self._episode_collision_flags = None
        log_dir = self.logger.get_dir()
        self._writer = SummaryWriter(log_dir) if log_dir else None
    
    # _on_rollout_start is called at the beginning of each rollout collection phase, 
    # where we reset the metric values for the new rollout.
    def _on_rollout_start(self) -> None:
        self._metric_values = {key: [] for key in self.CONTINUOUS_KEYS}

    # _on_step is called at every step of the environment during rollout collection,
    # where we extract the relevant metrics from the info dict and store them for logging.
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        if self._episode_collision_flags is None or len(self._episode_collision_flags) != len(infos):
            self._episode_collision_flags = [0.0] * len(infos)

        for key in self.CONTINUOUS_KEYS:
            values = [float(info[key]) for info in infos if key in info]
            if values:
                self._metric_values[key].extend(values)

        for env_idx, info in enumerate(infos):
            if float(info.get("collision", 0.0)) > 0.0:
                self._episode_collision_flags[env_idx] = 1.0

            if env_idx < len(dones) and dones[env_idx]:
                self._record_episode_metrics(env_idx, info)

        return True

    # _on_rollout_end is called at the end of each rollout collection phase, 
    # where we compute the average of the collected metrics and log them to TensorBoard.
    def _on_rollout_end(self) -> None:
        for key, values in self._metric_values.items():
            if not values:
                continue

            self.logger.record(f"task/{key}", float(np.mean(values)))

    # _on_training_end is called once at the end of training, 
    # where we close the TensorBoard writer to ensure all data is flushed and resources are released.
    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.close()

    # _record_episode_metrics is a helper function that logs episode-level metrics to TensorBoard when an episode ends,
    def _record_episode_metrics(self, env_idx, info):
        if self._writer is None:
            return

        self._episode_count += 1
        for key in self.EPISODE_KEYS:
            if key == "collision":
                value = self._episode_collision_flags[env_idx]
            else:
                value = 1.0 if float(info.get(key, 0.0)) > 0.0 else 0.0

            self._writer.add_scalar(f"task_episode/{key}", value, self._episode_count)

        episode_info = info.get("episode")
        if episode_info:
            self._writer.add_scalar("task_episode/length", float(episode_info["l"]), self._episode_count)
            self._writer.add_scalar("task_episode/reward", float(episode_info["r"]), self._episode_count)

        self._episode_collision_flags[env_idx] = 0.0

# Training uses PyBullet DIRECT mode for speed. Rendering is kept out of the
# training loop except for the lightweight preview callback.
def make_env():
    # Old environment setup kept for reference.
    # env = husky_env_neo.HuskyChaserEnv(render_mode="direct")

    # DIRECT mode runs PyBullet without opening the GUI, which is much faster for training.
    return HuskyChaserEnv(render_mode="direct")

# Main PPO training pipeline: create the vectorized environment, optionally
# resume a saved policy, configure PPO for the vector observation policy, attach
# callbacks, then train and save the final chaser model.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Number of PPO training timesteps for this run.",
    )
    parser.add_argument(
        "--resume",
        "--resume-from",
        dest="resume_from",
        default=None,
        help="Optional PPO model zip to continue training from.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore any saved model and train from scratch.",
    )
    args = parser.parse_args()

    # Keep this at 1 for now because this environment uses PyBullet's global API.
    # Multiple parallel environments can interfere with each other unless every PyBullet call
    # is carefully tied to a separate physicsClientId.
    vec_env = make_vec_env(make_env, n_envs=1)

    # Old CNN/image observation wrapper kept for reference.
    # vec_env = VecTransposeImage(vec_env)

    # VecMonitor adds standard episode reward/length summaries around the custom
    # info metrics emitted by HuskyChaserEnv.
    vec_env = VecMonitor(vec_env)

    # Resume logic supports long experiments. By default it continues from the
    # final saved zip if present, unless --fresh-start is passed.
    auto_resume_path = Path(FINAL_MODEL_PATH).with_suffix(".zip")
    resume_from = args.resume_from
    if resume_from is None and not args.fresh_start and auto_resume_path.exists():
        resume_from = str(auto_resume_path)

    # If the user has specified a model to resume from, we load that model and continue training.
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Could not find model to resume from: {resume_path}")

        # This is continued training/fine-tuning. It reuses the learned PPO
        # policy, then keeps improving it in the current environment.
        print(f"Loading existing PPO model from {resume_path}...")
        model = PPO.load(resume_path, env=vec_env, tensorboard_log=TENSORBOARD_LOG_DIR)
    else:
        model = PPO(
            # Old CNN/image policy kept for reference.
            # "CnnPolicy",

            # MlpPolicy matches the compact 8-value observation vector from
            # perception.py. The old CnnPolicy path is left above for reference.
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            learning_rate=3e-4,
            # Original PPO rollout size kept for reference.
            # n_steps=2048,
            # batch_size=128,

            # Smaller rollouts give quicker feedback on a CPU laptop while
            # still giving PPO enough samples to estimate pursuit advantages.
            n_steps=512,
            batch_size=64,
            gamma=0.99,      # Long enough horizon to value route planning.
            gae_lambda=0.95, # Stable advantage estimates for noisy physics.
            clip_range=0.2,  # Conservative PPO policy updates.
            ent_coef=0.02,   # Encourages exploration of turns/avoidance paths.
            vf_coef=0.5,     # Balances value learning with policy learning.
            max_grad_norm=0.5,
            device="auto",
        )

    # Checkpoints preserve intermediate policies every 10k steps. The custom
    # metrics and preview frame help explain training progress in the report.
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="husky_chaser_ppo",
    )
    task_metrics_callback = TaskMetricsCallback()

    base_env = vec_env.venv.envs[0]
    preview_callback = PreviewCallback(base_env, freq=256)
    callback = CallbackList([checkpoint_callback, task_metrics_callback, preview_callback])

    # Old eval callback kept for reference. We can add this back later with a separate eval env.
    # eval_callback = EvalCallback(
    #     vec_env,
    #     best_model_save_path="./best_model/",
    #     log_path="./logs/",
    #     eval_freq=5_000,
    #     n_eval_episodes=5,
    #     deterministic=True,
    # )

    # Start the PPO training loop, which will run for the specified number of timesteps. 
    # The callbacks will handle logging and checkpointing during training. 
    # At the end of training, the final model is saved to disk.
    print(f"Starting PPO training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="ppo_husky_chaser_mlp",
        )

        model.save(FINAL_MODEL_PATH)
        print(f"Training finished. Final model saved to {FINAL_MODEL_PATH}.zip")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
