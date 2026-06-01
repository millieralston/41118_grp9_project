"""
Demo script to visualize the trained TipBot model in action.
Run this to see the trained chaser robot pursue the runner in a visual PyBullet window.
"""

import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from husky_chaser_env import HuskyChaserEnv
import time
from ultralytics import YOLO
import cv2
import numpy as np

def run_demo(model_path="husky_chaser_ppo_final", num_episodes=5, steps_per_episode=1000):
    """
    Run trained model in visual mode.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        num_episodes: Number of episodes to run
        steps_per_episode: Max steps per episode
    """
    # Load the trained YOLO model
    yolo_model = YOLO("runs/detect/yolo_training/runner_detector_v2/weights/best.pt")

    print(f"Loading model from {model_path}...")
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
                
                # --- YOLO perception stream ---
                cam_img = env.get_camera_image_yolo()

                if step % 2 == 0:
                    results = yolo_model(cam_img)[0]
                
                annotated = results.plot()

                # make it easier to see
                annotated = cv2.resize(annotated, (960, 720))

                cv2.imshow("YOLO View (Demo)", annotated)
                cv2.waitKey(1)
                
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
        p.disconnect()
        print("Environment closed.")

if __name__ == "__main__":
    run_demo()
