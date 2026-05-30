import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from husky_chaser_env import HuskyChaserEnv

# Old CNN/image-based imports kept for reference.
# import gymnasium as gym
# from stable_baselines3.common.vec_env import VecTransposeImage
# from stable_baselines3.common.callbacks import EvalCallback
# import husky_env_neo   # Old filename from lier version; not in this repository.


# Longer training kept for reference.
# TOTAL_TIMESTEPS = 200_000

# PPO needs more than a smoke-test run to learn stable pursuit. Lower this if
# you only want to verify that the pipeline starts.
TOTAL_TIMESTEPS = 120_000
TENSORBOARD_LOG_DIR = "./husky_ppo_logs/"
CHECKPOINT_DIR = "./checkpoints/"
FINAL_MODEL_PATH = "husky_chaser_ppo_final"
MODEL_DIR = Path("models")


def make_env():
    # Old environment setup kept for reference.
    # env = husky_env_neo.HuskyChaserEnv(render_mode="direct")

    # DIRECT mode runs PyBullet without opening the GUI, which is much faster for training.
    return HuskyChaserEnv(render_mode="direct")


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

    vec_env = VecMonitor(vec_env)

    auto_resume_path = Path(FINAL_MODEL_PATH).with_suffix(".zip")
    resume_from = args.resume_from
    if resume_from is None and not args.fresh_start and auto_resume_path.exists():
        resume_from = str(auto_resume_path)

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

            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            learning_rate=3e-4,
            # Original PPO rollout size kept for reference.
            # n_steps=2048,
            # batch_size=128,

            # Smaller rollouts give quicker feedback on a CPU laptop.
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02, #increasing, increases exploration, but can lead to more unstable training. Adjust as needed.
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="husky_chaser_ppo",
    )

    # Old eval callback kept for reference. We can add this back later with a separate eval env.
    # eval_callback = EvalCallback(
    #     vec_env,
    #     best_model_save_path="./best_model/",
    #     log_path="./logs/",
    #     eval_freq=5_000,
    #     n_eval_episodes=5,
    #     deterministic=True,
    # )

    print(f"Starting PPO training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            tb_log_name="ppo_husky_chaser_mlp",
        )

        model.save(FINAL_MODEL_PATH)
        print(f"Training finished. Final model saved to {FINAL_MODEL_PATH}.zip")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
