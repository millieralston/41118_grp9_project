"""
Demo script to visualize the trained TipBot model in action.
Run this to see the trained chaser robot pursue the runner in a visual PyBullet window.
"""

import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from husky_chaser_env import HuskyChaserEnv
import time
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
import cv2
import numpy as np
from perception_yolo import get_yolo_runner

def run_demo(model_path="husky_chaser_ppo_final", num_episodes=5, steps_per_episode=1000, record=False, out_path=None):
    """
    Run trained model in visual mode.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        num_episodes: Number of episodes to run
        steps_per_episode: Max steps per episode
    """
    # Load the trained YOLO model (optional)
    yolo_model = None
    if YOLO is not None:
        try:
            yolo_model = YOLO("runs/detect/yolo_training/runner_detector_v2/weights/best.pt", verbose=False)
        except Exception:
            print("Warning: YOLO weights not found or failed to load; running demo without YOLO annotations.")
            yolo_model = None
    else:
        print("ultralytics package not available; running demo without YOLO annotations.")

    print(f"Loading model from {model_path}...")
    writer = None
    if record:
        import os as _os
        _os.makedirs("recordings", exist_ok=True)
        if out_path is None:
            out_path = f"recordings/demo_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 25.0, (960, 720))

    try:
        model = PPO.load(model_path)
    except:
        print(f"Error: Could not load model at {model_path}")
        print("Make sure you've trained the model first by running 'python train.py'")
        return
    
    print("Creating environment with PyBullet GUI...")
    env = HuskyChaserEnv(render_mode="gui")
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Get action from trained model
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                
                # --- YOLO perception stream (optional) ---
                cam_img = cv2.cvtColor(env.get_camera_image_yolo(), cv2.COLOR_RGB2BGR)
                annotated = None
                if yolo_model is not None:
                    try:
                        results = yolo_model(cam_img, verbose=False)[0]
                        annotated = results.plot()
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                    except Exception:
                        annotated = None

                if annotated is None:
                    # Fallback: show raw camera image if YOLO not available
                    try:
                        annotated = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                    except Exception:
                        annotated = np.zeros((480, 640, 3), dtype=np.uint8)

                runner_x, runner_y = get_yolo_runner(env)

                cv2.putText(
                    annotated,
                    f"x={runner_x:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # make it easier to see
                annotated = cv2.resize(annotated, (960, 720))

                cv2.imshow("YOLO View (Demo)", annotated)
                cv2.waitKey(1)

                if writer is not None:
                    # Write BGR frame (annotated from Ultralytics is display-ready)
                    writer.write(annotated)
                
                # Small delay for visual clarity
                time.sleep(0.01)
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1} finished. Total Reward: {episode_reward:.2f}")
        
        print("\nDemo completed! Close the PyBullet window to exit.")
        # Keep window open
        input("Press Enter to close...")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved demo video to {out_path}")
        p.disconnect()
        print("Environment closed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="husky_chaser_ppo_final", help="Model path (without .zip)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--record", action="store_true", help="Record annotated demo to MP4")
    parser.add_argument("--out", default=None, help="Output file path for recording")

    args = parser.parse_args()
    run_demo(model_path=args.model, num_episodes=args.episodes, steps_per_episode=args.steps, record=args.record, out_path=args.out)
