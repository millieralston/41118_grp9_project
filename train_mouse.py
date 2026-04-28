from stable_baselines3 import PPO
from maze_gen import MouseMazeEnv
import os

TOTAL_TIMESTEPS = 500000

def train():
    env = MouseMazeEnv(renders=True) 
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_maze_tensorboard/")
    model.learn(total_timesteps=500000)
    
    if os.path.exists("mouse_model.zip"):
        print("Loading existing model...")
        model = PPO.load("mouse_model", env=env)
    else:
        model = PPO("MlpPolicy", env, 
                    learning_rate=0.0003, 
                    n_steps=512, 
                    batch_size=256, 
                    verbose=1)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("mouse_model")
    print("Model saved.")

if __name__ == "__main__":
    train()