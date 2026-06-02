# TipBot AI Project

TipBot is a group project from UTS AI in Robotics (Group 9) that combines reinforcement learning and vision-based perception to train a Husky robot to chase a second Husky target inside a PyBullet environment.

## Project summary

This repository contains a simulated autonomous pursuit system in which:

- a **chaser** Husky robot is controlled by a PPO agent,
- a **runner** Husky robot follows a scripted escape policy,
- the environment includes randomly placed obstacles inside a bounded arena,
- a CNN-based visual pipeline is used to detect the runner from the chaser's onboard camera.

The main goal is to learn a robust pursuit policy that intercepts the runner while avoiding obstacles, using shaped rewards and a compact perception vector.

## Repository structure

- `husky_chaser_env.py`: main gym environment variant for the pursuit task
- `train.py`: PPO training script for the chaser agent
- `demo.py`: visual demo runner, now with optional video recording
- `collect_yolo_dataset.py`: dataset collection script for YOLO training
- `train_yolo.py`: YOLO model training script
- `yolo_live_test.py`: live YOLO inference test in the simulation
- `perception.py`: observation extraction and perception utilities
- `gui.py`: optional GUI components for project interaction
- `checkpoints/`: saved PPO checkpoints
- `husky_ppo_logs/`: TensorBoard training logs

## Setup

1. Activate the repository virtual environment:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available or incomplete, install the main packages:

```bash
pip install gymnasium pybullet stable-baselines3 opencv-python ultralytics torch torchvision
```

> Note: `ultralytics` is only required for YOLO annotation in the demo. The demo can still run without it, but it will show raw camera frames instead of annotated detections.

## Training the PPO agent

Train the RL agent using:

```bash
python train.py --timesteps 120000
```

This script:

- creates the `HuskyChaserEnv` in `direct` mode for fast training,
- uses Stable Baselines3 PPO with an MLP policy,
- saves checkpoints every 10,000 timesteps,
- writes a final model to `husky_chaser_ppo_final.zip`.

To resume training from a saved model:

```bash
python train.py --resume husky_chaser_ppo_final.zip
```

To force training from scratch:

```bash
python train.py --fresh-start --timesteps 120000
```

## Running the demo

Run the trained model in a PyBullet GUI:

```bash
python demo.py --model husky_chaser_ppo_final --episodes 5
```

To record the demo to an MP4 file:

```bash
python demo.py --model husky_chaser_ppo_final --episodes 5 --record --out recordings/demo_run.mp4
```

This demo captures the live camera view from the chaser and saves a video of the running episode. If `ultralytics` is installed and YOLO weights are available, the recorded video will include annotated detection boxes.

## YOLO dataset and detection

The project includes a YOLO perception path that is currently used for demonstrations and future integration:

- `collect_yolo_dataset.py` captures simulated camera frames and segmentation-based labels for the runner.
- `train_yolo.py` fine-tunes a YOLO detector on the collected dataset.
- `yolo_live_test.py` runs live YOLO inference in the PyBullet environment.

The demo script expects YOLO weights at:

```bash
runs/detect/yolo_training/runner_detector_v2/weights/best.pt
```

If those weights are missing, the demo will still run without YOLO annotations.

## Notes

- The current training pipeline uses a compact observation vector from `perception.py`, not the raw 64×64 image directly.
- The environment randomizes obstacles, runner spawn, and chaser spawn to improve generalization.
- The reward is shaped to encourage progress toward the runner, penalize collisions, and reward capture with a large terminal bonus.

## Useful commands

```bash
source venv/bin/activate
python train.py --timesteps 120000
python demo.py --model husky_chaser_ppo_final --episodes 5
python demo.py --model husky_chaser_ppo_final --episodes 5 --record --out recordings/demo_run.mp4
```

## Citation

If you use this work or adapt it in your own project, please cite:

```bibtex
@misc{tipbot2026,
  title={TipBot: Reinforcement Learning for Robotic Target Interception},
  author={Melroop and Millie and Eliza},
  year={2026},
  institution={UTS, AI in Robotics, Group 9}
}
```
