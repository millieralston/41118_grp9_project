import time
from stable_baselines3 import PPO
from maze_gen import MouseMazeEnv

def test():
    env = MouseMazeEnv(renders=True)
    model = PPO.load("mouse_model", env=env)

    obs, info = env.reset()
    done = False
    total_reward = 0

    print("Testing Mouse AI...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(1./60.) # Slow down for visualization
    
    print(f"Finished! Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test()