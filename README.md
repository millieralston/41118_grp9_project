# TipBot AI Project

Hello everyone, we are Group 9. Our team member names are Melroop, Millie and Eliza. Our AI in robotics project is called the TipBot.

## Project summary

Our system is designed to simulate a reinforcement learning system, where two Husky bots are loaded within a PyBullet simulated environment to chase another Husky bot while simultaneously avoiding obstacles.

The TipBot system trains a robotic UGV (unmanned ground vehicle) to follow and intercept ground targets. Through this project, we hope to better demonstrate our understanding of the topics learned throughout this course.

## Project structure

- **41118_grp9_project/**: Group project involving Husky robot environment and perception
  - `husky_env.py`: Main environment file for Husky robot simulation
  - `husky_chaser_env.py`: Environment variant for chasing behavior
  - `perception.py`: Perception module for robot vision
  - `train.py`: Training script
  - `husky.urdf`: Robot model definition
  - `checkpoints/`: Saved model checkpoints

## Current progress

Currently, we have developed a fully built environment where two Husky robots are initialized on the ground at random locations. The environment contains various obstacles that are initialized using PyBullet at random positions, and a boundary is created around the arena to confine the robots.

After the Husky bots are loaded into the environment, the runner robot follows a scripted policy to evade the chaser robot, while the chaser (which is our subject for the training model) pursues and intercepts the runner.

The reward functions have also been implemented, including both dense and sparse signals:

- +200 for catching the runner
- −0.8 × distance per step to encourage closing the gap
- −5.0 obstacle/wall proximity penalty
- −5.0 stuck penalty (robot not moving) to discourage collisions

## Challenges

During development, we encountered a few challenges:

- Version conflicts between group members caused inconsistencies when running PyBullet, specifically regarding displaying the camera image data.
- Consequently, we updated the code to be more robust by not assuming returned data formats.
- We’ve had to adapt our code for training the Husky robot to use a dedicated GPU, as given that we are using a CNN, this shouldn’t be run on a CPU.
- Designing the observation space also needed considerable thought, as we needed to provide enough useful information about the environment without overwhelming it and keep the state representation manageable.
- We’ve been attempting to train the robot to properly evade obstacles while actively chasing and running, but the agent’s policy has yet to reach convergence or behave appropriately.
- The first trial resulted in the chaser robot not moving at all.

## Next steps

- More robust reward policy, denser.
- Runner with better scripted movement (it currently can collide).
- Better collision avoidance for both Husky bots.
- CNN-based policy fully implemented.

The key difference from assignment 2 that we hope to implement is the addition of CNN image processing. The chaser robot uses a forward-facing camera to capture a 64×64 pixel image of what is directly in front of it (this is its only window into the environment). To improve speed, we can add things like multi-frame stepping and parallel training.

So instead of hand-coding rules like "if an obstacle is nearby, turn left," we give the raw image to a Convolutional Neural Network. The CNN learns to extract meaningful spatial features from the image on its own, such as edges, shapes, and the runner’s position.

## Technical Inputs

### Observation

- Camera image: 64 × 64 × 3 RGB
- Camera FOV: 60°
- Projection near/far clip: 0.1 / 100.0

### Action

- 2 continuous controls: [linear_velocity, angular_velocity]
- Scaled in env:
  - linear: action[0] * 4.0
  - angular: action[1] * 2.0

### Simulation / Frame Rate

- PyBullet stepSimulation() per action step
- Demo loop uses time.sleep(1./240.) → 240 Hz step rate
- Training uses DIRECT mode, so step rate is uncapped and runs as fast as possible

### Episode setup

- 12 random trees
- 5 random benches
- Random spawn positions for chaser and runner inside boundary ±10

## PPO training configuration

- Vectorized environments: n_envs = 4
- Policy: CnnPolicy
- n_steps = 2048
- batch_size = 128
- gamma = 0.99
- gae_lambda = 0.95
- clip_range = 0.2
- ent_coef = 0.01
- vf_coef = 0.5
- max_grad_norm = 0.5
- Total timesteps: 200,000

## Logging / checkpoints

- TensorBoard logs: ./husky_ppo_logs/
- Best model eval: every 5000 steps
- Checkpoints: every 10000 steps

## Running the GUI

python gui.py

## Running the HTML Documentation Site

To view the project documentation locally:

```bash
cd ~/41118_grp9_project
python3 -m http.server 8000
```

Then open your browser and navigate to:
- Main site: http://localhost:8000
- Home page: http://localhost:8000/index.html
- Project page: http://localhost:8000/project.html
- README page: http://localhost:8000/readme.html
- Documentation: http://localhost:8000/docs/index.html

## Citation

If you use this project in your research or work, please cite as follows:

```
@misc{tipbot2026,
  title={TipBot: Reinforcement Learning for Robotic Target Interception},
  author={Melroop and Millie and Eliza},
  year={2026},
  institution={UTS, AI in Robotics, Group 9}
}
```

Documentation template acquired from

@article{park2021nerfies
  author    = {Park, Keunhong and Sinha, Utkarsh and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Seitz, Steven M. and Martin-Brualla, Ricardo},
  title     = {Nerfies: Deformable Neural Radiance Fields},
  journal   = {ICCV},
  year      = {2021},
}
