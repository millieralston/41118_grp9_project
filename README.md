# TipBot AI Project

TipBot is a group project from UTS 41118 AI in Robotics (Group 9) that combines reinforcement learning and vision-based perception to train a Husky robot to chase a second Husky target inside a PyBullet simulation.

**Authors:** Millie Ralston, Eliza Tam, Melroop Nijjar

## Project Summary

This repository contains a simulated autonomous pursuit system in which:

- a **chaser** Husky robot is controlled by a PPO agent,
- a **runner** Husky robot follows a scripted evasion policy,
- the environment includes randomly placed obstacles inside a bounded arena,
- an optional YOLOv11 visual pipeline detects the runner from the chaser's onboard camera.

The chaser learns a robust pursuit policy using shaped rewards and a compact **8-dimensional** observation vector derived from simulation state: the runner's position and the three nearest obstacles, all expressed in the chaser's local frame.

## Repository Structure

| File | Description |
|------|-------------|
| `husky_chaser_env.py` | Main Gymnasium environment: PPO training, scripted runner, obstacle/physics setup |
| `husky_env.py` | Earlier, simpler environment variant (kept for reference) |
| `train.py` | PPO training script (MLP policy, TensorBoard logging, checkpointing) |
| `demo.py` | Visual demo with optional MP4 recording and YOLO bounding-box overlay |
| `gui.py` | Tkinter GUI: start/stop training and demo, live reward chart, TF log viewer |
| `perception.py` | 8-D observation: runner position + 3 nearest obstacles in chaser frame |
| `perception_yolo.py` | Experimental 10-D observation: replaces ground-truth runner coords with YOLO detection output (includes `runner_visible` and `runner_confidence` channels) |
| `collect_yolo_dataset.py` | Captures labelled camera frames from simulation for YOLO training |
| `split_dataset.py` | Train/val dataset split utility |
| `train_yolo.py` | Fine-tunes YOLOv11n on the collected dataset |
| `visualise_yolo.py` | Visualises YOLO bounding-box labels on dataset images |
| `yolo_live_test.py` | Live YOLO inference inside a running PyBullet environment |
| `test.py` / `test_yolo.py` | Development test scripts |
| `husky_chaser_ppo_final.zip` | Final trained PPO model |
| `husky.urdf` | Husky robot URDF model |
| `dataset.yaml` | YOLO dataset configuration |
| `yolo11n.pt` | Base YOLOv11n weights used as starting point for fine-tuning |
| `yolo26n.pt` | Alternative YOLO checkpoint |
| `index.html` | Project webpage |
| `checkpoints/` | PPO step checkpoints saved every 10,000 timesteps during training |
| `husky_ppo_logs/` | TensorBoard training logs |

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
pip install gymnasium pybullet stable-baselines3 opencv-python ultralytics torch torchvision tensorboard matplotlib
```

> `ultralytics` is required for YOLO annotations in the demo and for training the detector. The PPO demo runs without it but shows raw camera frames instead.

## Training the PPO Agent

```bash
python train.py --timesteps 120000
```

This script:
- creates `HuskyChaserEnv` in `direct` mode for fast training,
- uses Stable Baselines3 PPO with an MLP policy (lr=3e-4, n_steps=512, batch=64),
- saves checkpoints every 10,000 timesteps to `checkpoints/`,
- logs task metrics (distance, capture rate, collisions) to TensorBoard,
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

If YOLO weights are present at `runs/detect/yolo_training/runner_detector_v2/weights/best.pt`, the recording will include annotated detection boxes. Otherwise the raw camera feed is shown.

## GUI Launcher

```bash
python gui.py
```

The Tkinter control panel lets you:
- **Start / Stop Training** — launches `train.py` as a subprocess and streams its console output,
- **Run / Stop Demo** — launches `demo.py` in a separate window,
- **Open TensorBoard** — starts TensorBoard on `http://localhost:6006`,
- **Live Training chart** — plots mean episode reward vs timesteps in real time,
- **TF Log Viewer** — loads and displays the six key TensorBoard scalars (reward, length, policy loss, value loss, entropy, explained variance) from any saved run.

## YOLO Dataset and Detection

```bash
# Collect labelled camera frames from simulation
python collect_yolo_dataset.py

# Split into train/val
python split_dataset.py

# Fine-tune YOLOv11n on the collected dataset
python train_yolo.py

# Test live inference inside the simulation
python yolo_live_test.py
```

`collect_yolo_dataset.py` targets 3,500 samples, filters out frames where the runner is heavily occluded or covers fewer than 200 pixels, and writes YOLO-format labels alongside each image. `train_yolo.py` fine-tunes `yolo11n.pt` for up to 100 epochs with early stopping (patience=20) and saves the best weights to `yolo_training/runner_detector_v2/`.

## Reported Evaluation Results

Evaluated on `husky_chaser_ppo_final.zip` over 50 episodes:

| Metric | Value |
|--------|-------|
| Capture Rate | 72.0% |
| Mean Distance to Runner | 8.73 m |
| Collision Rate | 44.0% |
| Mean Episode Length | 491.3 steps |
| Mean Episode Reward | 1341.0 |

| Statistic              | Min  | Mean   | Max    | Std    |
|------------------------|------|--------|--------|--------|
| Episode Length (steps) | 113  | 491.3  | 1000   | 239.2  |
| Distance to Runner (m) | 2.23 | 8.73   | 20.81  | 4.09   |
| Reward per Episode | −1316.2  | 1341.0 | 2805.7 | 1091.1 |

## Reward Function

The per-step reward has several components. Key values:

| Signal | Value |
|--------|-------|
| Catch bonus (contact or distance ≤ 2.3 m) | +800 |
| Closing distance (progress per step) | +60 × progress |
| Moving away from runner | −12 × \|loss\| |
| Close-range bonus (within 5 m) | +1.8 × (5 − distance) |
| Obstacle contact (growing with consecutive steps) | −3.0 to −4.5 per step |
| Wall contact | −4.0 per step |
| Wall boundary crossed (inner margin) | −5.0 |
| Stuck terminal (80 consecutive stuck steps) | −35 |
| Obstacle contact terminal (45 consecutive steps) | −35 |
| Flip terminal (roll or pitch > 1.0 rad) | −100 |
| Step cost (increasing over episode) | −0.01 to −0.025 |

## Viewing TensorBoard Logs

```bash
tensorboard --logdir ./husky_ppo_logs/
```

Then open `http://localhost:6006`.

## Citation

```bibtex
@misc{tipbot2026,
  title={TipBot: AI AGENT FOR GROUND BASED TARGET INTERCEPTION},
  author={Millie Ralston and Eliza Tam and Melroop Nijjar},
  year={2026},
  institution={UTS, 41118 AI in Robotics, Group 9}
}
```
