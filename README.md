# TipBot AI Project

TipBot is a group project from UTS 41118 AI in Robotics (Group 9) that combines reinforcement learning and vision-based perception to train a Husky robot to chase a second Husky target inside a PyBullet simulation.

**Authors:** Millie Ralston, Eliza Tam, Melroop Nijjar

## Project Summary

This repository contains a simulated autonomous pursuit system in which:

- a **chaser** Husky robot is controlled by a PPO agent,
- a **runner** Husky robot follows a scripted evasion policy,
- the environment includes randomly placed obstacles inside a bounded arena,
- a YOLOv11 visual pipeline detects the runner from the chaser's onboard camera.

The chaser learns a robust pursuit policy using shaped rewards and a compact 10-dimensional perception vector derived from both simulation state and YOLO detections.

## Repository Structure

| File | Description |
|------|-------------|
| `husky_chaser_env.py` | Main Gymnasium environment for the pursuit task |
| `husky_env.py` | Alternate environment variant |
| `train.py` | PPO training script (MLP policy) |
| `demo.py` | Visual demo with optional video recording and YOLO overlay |
| `perception.py` | Observation extraction: runner position + 3 nearest obstacles |
| `perception_yolo.py` | YOLO-based runner detection perception module |
| `gui.py` | GUI launcher for training monitoring |
| `collect_yolo_dataset.py` | Simulated camera frame capture for YOLO dataset |
| `split_dataset.py` | Train/val dataset split utility |
| `train_yolo.py` | Fine-tunes YOLOv11n on collected dataset |
| `yolo_live_test.py` | Live YOLO inference inside PyBullet |
| `visualise_yolo.py` | Visualises YOLO bounding box labels on dataset images |
| `evaluate_policies.py` | Runs trained policy for N episodes and collects metrics |
| `extract_training_metrics.py` | Extracts convergence metrics from TensorBoard logs |
| `run_metrics_collection.sh` | Shell script to run full metrics pipeline |
| `test.py` / `test_yolo.py` | Development test scripts |
| `checkpoints/` | PPO step checkpoints saved during training |
| `husky_ppo_logs/` | TensorBoard training logs |
| `husky_chaser_ppo_final.zip` | Final trained model |
| `husky.urdf` | Husky robot URDF model |
| `dataset.yaml` | YOLO dataset configuration |

## Setup

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages:

```bash
pip install gymnasium pybullet stable-baselines3 opencv-python ultralytics torch torchvision tensorboard
```

> `ultralytics` is required for YOLO annotations in the demo. The demo runs without it but shows raw camera frames instead.

## Training the PPO Agent

```bash
python train.py --timesteps 120000
```

This script:
- creates `HuskyChaserEnv` in `direct` mode for fast training,
- uses Stable Baselines3 PPO with an MLP policy,
- saves checkpoints every 10,000 timesteps to `checkpoints/`,
- writes the final model to `husky_chaser_ppo_final.zip`.

**Resume from a checkpoint:**

```bash
python train.py --resume husky_chaser_ppo_final.zip
```

**Force fresh training:**

```bash
python train.py --fresh-start --timesteps 120000
```

## Running the Demo

```bash
python demo.py --model husky_chaser_ppo_final --episodes 5
```

**Record to MP4:**

```bash
python demo.py --model husky_chaser_ppo_final --episodes 5 --record --out recordings/demo_run.mp4
```

If YOLO weights are present at `runs/detect/yolo_training/runner_detector_v2/weights/best.pt`, the recording will include annotated detection boxes.

## YOLO Dataset and Detection

```bash
# Collect dataset frames from simulation
python collect_yolo_dataset.py

# Split into train/val
python split_dataset.py

# Train YOLOv11n detector
python train_yolo.py

# Test live inference
python yolo_live_test.py
```

## Evaluating the Policy

Run evaluation across 50 episodes and save results to JSON:

```bash
python evaluate_policies.py --episodes 50
```

Results are written to `evaluation_results.json`.

**With GUI rendering:**

```bash
python evaluate_policies.py --episodes 50 --render
```

**Full metrics pipeline (evaluation + TensorBoard extraction):**

```bash
bash run_metrics_collection.sh 50
```

## Evaluation Results

Evaluated on `husky_chaser_ppo_final.zip` over 50 episodes:

| Metric | Value |
|--------|-------|
| Capture Rate | 72.0% |
| Mean Distance to Runner | 8.73 m |
| Collision Rate | 44.0% |
| Mean Episode Length | 491.3 steps |
| Mean Episode Reward | 1341.0 |

| Statistic | Min | Mean | Max | Std |
|-----------|-----|------|-----|-----|
| Episode Length (steps) | 113 | 491.3 | 1000 | 239.2 |
| Distance to Runner (m) | 2.23 | 8.73 | 20.81 | 4.09 |
| Reward per Episode | −1316.2 | 1341.0 | 2805.7 | 1091.1 |

## Reward Function

| Signal | Value |
|--------|-------|
| Catch bonus | +800 |
| Distance shaping (per step) | −0.8 × distance |
| Obstacle/wall proximity | −5.0 |
| Stuck penalty | −5.0 |

## Viewing TensorBoard Logs

```bash
tensorboard --logdir ./husky_ppo_logs/
```

Then open `http://localhost:6006`.

## Citation

```bibtex
@misc{tipbot2026,
  title={TipBot: Reinforcement Learning for Robotic Target Interception},
  author={Millie Ralston and Eliza Tam and Melroop Nijjar},
  year={2026},
  institution={UTS, 41118 AI in Robotics, Group 9}
}
```
