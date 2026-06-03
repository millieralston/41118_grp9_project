import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from husky_chaser_env import HuskyChaserEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="husky_chaser_ppo_final.zip",
        help="Path to the trained PPO model zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes to watch.",
    )
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print the model action every 30 steps while testing.",
    )
    parser.add_argument(
        "--runner-spawn-mode",
        choices=("random", "front"),
        default="random",
        help="Use random PPO-style runner spawns or force the runner into the chaser camera view.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train the model first with: python train.py"
        )

    env = HuskyChaserEnv(render_mode="gui", runner_spawn_mode=args.runner_spawn_mode)
    model = PPO.load(model_path)

    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            step_count = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if args.print_actions and step_count % 30 == 0:
                    print(f"step={step_count} action={action}")
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step_count += 1
                time.sleep(1.0 / 240.0)

            print(f"Episode {episode} finished with reward {total_reward:.2f}")
    except KeyboardInterrupt:
        print("Test stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
