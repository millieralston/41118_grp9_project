import contextlib
import gymnasium as gym
import os
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import sys
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from perception import create_observation_space, get_observation, get_relative_position

# this is to suppress the noisy URDF importer warnings while keeping normal training logs visible.
@contextlib.contextmanager
def suppress_pybullet_load_warnings():
    """Hide noisy URDF importer warnings while keeping normal training logs visible."""
    saved_stdout = None
    saved_stderr = None
    devnull = None

    # PyBullet's URDF loader can be very verbose with warnings about missing optional features.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        if saved_stdout is not None:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)
        if saved_stderr is not None:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        if devnull is not None:
            os.close(devnull)


class HuskyChaserEnv(gym.Env):
    def __init__(self, render_mode="direct", runner_spawn_mode="random"): # "gui" for visualization, "direct" for fast training
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Old CNN/image observation space kept for reference.
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # MLP/PPO uses a small vector from perception.py instead of camera pixels.
        self.observation_space = create_observation_space()
        
        # The play area is a 20x20 square, but the observation space only includes relative positions of the runner and obstacles, 
        # so the boundary is effectively smaller. The chaser can see obstacles up to about 5 units away, 
        # but the environment extends beyond that to allow for more natural pursuit and evasion.
        self.boundary = 10
        self.obstacle_positions = []
        self.obstacle_body_ids = []
        self.wall_body_ids = []
        self.obstacle_line_ids = []
        self.obstacle_text_ids = []

        self.render_mode = render_mode.lower()
        self.debug_perception = self.render_mode == "gui"
        if runner_spawn_mode not in {"random", "front"}:
            raise ValueError("runner_spawn_mode must be 'random' or 'front'")
        self.runner_spawn_mode = runner_spawn_mode

        # Longer episodes give the chaser time to route around obstacles,
        # reacquire the runner, and finish the catch.
        self.max_steps = 1000
        self.step_count = 0

        # Movement tuning values. These are wheel target velocities, not m/s.
        # The top speeds are high enough for visible pursuit, while acceleration
        # and turn scaling are kept conservative to prevent wheelies.
        self.chaser_min_forward_speed = 1.5
        self.chaser_max_forward_speed = 22.5
        self.chaser_angular_scale = 5.5
        self.runner_min_forward_speed = 3.5
        self.runner_max_forward_speed = 13.0
        self.runner_turn_gain = 4.0
        self.runner_max_turn_speed = 7.0
        self.max_wheel_velocity = 28.0
        self.max_wheel_accel = 3.0
        self.motor_force = 90.0
        self.sim_steps_per_env_step = 4
        self.downforce_scale = 12.0
        self.pitch_roll_damping = 42.0
        self.wheel_joint_indices = [2, 3, 4, 5]
        self.current_wheel_targets = {}

        self.prev_distance = None
        self.prev_chaser_xy = None
        self.stuck_steps = 0
        self.obstacle_contact_steps = 0
        self.catch_distance = 2.3
        self.near_distance = 5.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_avoidance_turn = 0.0
        self.last_target_assist_strength = 0.0
        self.last_front_obstacle_distance = None
        self.escape_steps = 0
        self.escape_turn_direction = 1.0
        self.avoid_commit_steps = 0
        self.avoid_turn_direction = 1.0
        # self.runner_ang = 0.0
        # self.runner_turn_timer = 0
        
        # Initialising IDs for observation space debug visuals
        self.runner_line_id = -1
        self.runner_text_id = -1

        self.obstacle_line_ids = []
        self.obstacle_text_ids = []

        # Load YOLO model when available. PPO training/testing still works without it.
        self.yolo_model = None
        if YOLO is not None:
            try:
                self.yolo_model = YOLO("runs/detect/yolo_training/runner_detector_v2/weights/best.pt", verbose=False)
            except Exception:
                self.yolo_model = None

        self.last_runner_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.runner_visible = False

        self.yolo_cache = np.zeros(2, dtype=np.float32)
        self.yolo_step_counter = 0
        self.yolo_skip = 2  # run YOLO every 2 steps (start small)

        # Handle GUI vs DIRECT mode
        if self.render_mode == "gui":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)   # For training (fast)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # the reset function initializes the environment, placing the chaser and runner in valid positions and setting up the obstacles. 
    # It returns the initial observation and an empty info dictionary.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        runner_spawn_mode = options.get("runner_spawn_mode", self.runner_spawn_mode)
        if runner_spawn_mode not in {"random", "front"}:
            raise ValueError("runner_spawn_mode must be 'random' or 'front'")

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=80,
            enableConeFriction=1,
        )
        self.obstacle_positions = []
        self.obstacle_body_ids = []
        self.wall_body_ids = []
        self.obstacle_line_ids = []
        self.obstacle_text_ids = []
        self.step_count = 0
        self.prev_distance = None
        self.prev_chaser_xy = None
        self.stuck_steps = 0
        self.obstacle_contact_steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_avoidance_turn = 0.0
        self.last_target_assist_strength = 0.0
        self.last_front_obstacle_distance = None
        self.escape_steps = 0
        self.escape_turn_direction = 1.0
        self.avoid_commit_steps = 0
        self.avoid_turn_direction = 1.0
        self.current_wheel_targets = {}

        # initialising runner visibility and confidence for observation space
        self.runner_visible = False
        self.runner_confidence = 0.0
        self.last_runner_xy = np.array([0.0, 0.0], dtype=np.float32)
        
        self.runner_line_id = -1
        self.runner_text_id = -1
        self.obstacle_line_ids = []
        self.obstacle_text_ids = []
        
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=1.4,
            rollingFriction=0.002,
            spinningFriction=0.002,
            restitution=0.0,
        )
        self._build_fence()

        for _ in range(12):
            self._create_tree([random.uniform(-8, 8), random.uniform(-8, 8)])
        for _ in range(5):
            self._create_bench([random.uniform(-7, 7), random.uniform(-7, 7)])

        chaser_pos = self._get_valid_spawn()

        yaw = np.random.uniform(-np.pi, np.pi)
        quat = p.getQuaternionFromEuler([0, 0, yaw])

        with suppress_pybullet_load_warnings():
            self.chaser = p.loadURDF("husky/husky.urdf", basePosition=chaser_pos, baseOrientation=quat)
        p.changeVisualShape(self.chaser, -1, rgbaColor=[1, 0, 0, 1])
        self._configure_husky_dynamics(self.chaser)
        
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)

        if runner_spawn_mode == "front":
            runner_pos = self._get_front_runner_spawn(chaser_pos, chaser_orn)
        else:
            runner_pos = self._get_valid_spawn(reference_positions=[chaser_pos], min_distance=5.0)

        with suppress_pybullet_load_warnings():
            self.runner = p.loadURDF("husky/husky.urdf", basePosition=runner_pos)
        p.changeVisualShape(self.runner, -1, rgbaColor=[0, 0, 1, 1])
        self._configure_husky_dynamics(self.runner)

        self.prev_distance = np.linalg.norm(
            np.array(chaser_pos[:2]) - np.array(runner_pos[:2])
        )
        self.prev_chaser_xy = np.array(chaser_pos[:2], dtype=np.float32)
        
        if self.render_mode == "gui":
            time.sleep(0.1)  # Small delay helps camera stability
        return self._get_obs(), {}

    # Obtain the obstacle and runner positons relative to the chaser, and return them as an observation vector.
    def _get_obs(self):
        # Old CNN/image observation kept for reference.
        # return self.get_camera_image()

        return get_observation(self)

    # this is the same relative position calculation used in perception.py, 
    # but included here for debug visualization of the observation space in the GUI. 
    # It draws lines from the chaser to the runner and obstacles, 
    # and prints their relative positions in the console.
    def get_camera_image(self):
        return self._get_camera_image(64, 64)

    def get_camera_image_yolo(self):
        return self._get_camera_image(640, 480)

    def get_preview_image(self):
        return self._get_camera_image(320, 240)

    def _get_camera_image(self, width=64, height=64):
        pos, orn = p.getBasePositionAndOrientation(self.chaser)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

        cam_pos = [
            pos[0] + forward[0] * 0.5,
            pos[1] + forward[1] * 0.5,
            pos[2] + 0.6,
        ]
        target = [
            pos[0] + forward[0] * 3.0,
            pos[1] + forward[1] * 3.0,
            pos[2] + 0.4,
        ]

        view_matrix = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=100.0,
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
        return rgb.astype(np.uint8)

    # step function applies the action to the chaser, moves the runner, steps the simulation,
    # and calculates the reward and termination status.
    def step(self, action):
        self.step_count += 1
        raw_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action = self._apply_chaser_obstacle_reflex(raw_action.copy())
        self.last_raw_action = raw_action
        self.last_action = action

        forward_signal = (float(action[0]) + 1.0) * 0.5
        chaser_linear_speed = (
            self.chaser_min_forward_speed
            + forward_signal * (self.chaser_max_forward_speed - self.chaser_min_forward_speed)
        )
        if float(action[0]) <= -0.95:
            if self.escape_steps > 0 or self.obstacle_contact_steps > 0 or self.stuck_steps > 8:
                chaser_linear_speed = -3.0
            else:
                chaser_linear_speed = 0.0
        chaser_angular_speed = float(action[1]) * self.chaser_angular_scale

        # Slow down slightly during sharp turns. This keeps high-speed pursuit
        # usable without generating enough torque to lift the front wheels.
        chaser_linear_speed *= 1.0 - 0.25 * abs(float(action[1]))
        runner_linear_speed, runner_angular_speed = self._scripted_runner_command()

        self._drive_husky(
            self.chaser,
            chaser_linear_speed,
            chaser_angular_speed,
        )
        self._drive_husky(
            self.runner,
            runner_linear_speed,
            runner_angular_speed,
        )
        for _ in range(self.sim_steps_per_env_step):
            self._apply_ground_stabilization(self.chaser)
            self._apply_ground_stabilization(self.runner)
            p.stepSimulation()
        
        obs = self._get_obs()
        reward, terminated, info = self._calculate_reward()
        truncated = self.step_count >= self.max_steps
        info["truncated"] = float(truncated)
        return obs, reward, terminated, truncated, info

    # ... keep your _build_fence, _create_tree, _create_bench, _get_valid_spawn, _drive_husky ...
    def _build_fence(self):
        """Creates a perimeter of boxes around the play area."""
        fence_height = 1.0
        thickness = 0.2
        color = [0.3, 0.2, 0.1, 1] # Wood brown
        
        # Positions and half-extents for the 4 walls [x, y, x_size, y_size]
        walls = [
            ([self.boundary, 0, fence_height/2], [thickness, self.boundary, fence_height/2]), # Right
            ([-self.boundary, 0, fence_height/2], [thickness, self.boundary, fence_height/2]), # Left
            ([0, self.boundary, fence_height/2], [self.boundary, thickness, fence_height/2]), # Top
            ([0, -self.boundary, fence_height/2], [self.boundary, thickness, fence_height/2]), # Bottom
        ]
        
        for pos, size in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
            body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
            p.changeDynamics(body_id, -1, lateralFriction=1.2, restitution=0.0)
            self.wall_body_ids.append(body_id)
            # self.obstacle_body_ids.extend(self.wall_body_ids)

    def _create_tree(self, pos):
        self.obstacle_positions.append(pos)
        trunk_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.5)
        trunk_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.5, rgbaColor=[0.5, 0.3, 0.1, 1])
        trunk_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=trunk_col, baseVisualShapeIndex=trunk_vis, basePosition=[pos[0], pos[1], 0.75])
        p.changeDynamics(trunk_id, -1, lateralFriction=1.2, restitution=0.0)
        self.obstacle_body_ids.append(trunk_id)
        
        leaf_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.8, rgbaColor=[0.1, 0.6, 0.2, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=leaf_vis, basePosition=[pos[0], pos[1], 1.8])

    def _create_bench(self, pos):
        self.obstacle_positions.append(pos)
        bench_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.15])
        bench_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.15], rgbaColor=[0.4, 0.3, 0.2, 1])
        bench_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bench_col, baseVisualShapeIndex=bench_vis, basePosition=[pos[0], pos[1], 0.15])
        p.changeDynamics(bench_id, -1, lateralFriction=1.2, restitution=0.0)
        self.obstacle_body_ids.append(bench_id)

    # This ensures the chaser and runner start in valid positions that are clear of obstacles 
    # and a certain distance apart from each other.
    def _get_valid_spawn(self, reference_positions=None, min_distance=0.0):
        """Ensures the robot starts clear of obstacles and optional references."""
        reference_positions = reference_positions or []
        while True:
            # Pick a random spot inside the fence
            pos = [random.uniform(-self.boundary + 1, self.boundary - 1), 
                   random.uniform(-self.boundary + 1, self.boundary - 1)]
            
            # Check distance against all placed obstacles (trees/benches)
            clear_of_obstacles = all(
                np.linalg.norm(np.array(pos) - np.array(obs)) > 1.2
                for obs in self.obstacle_positions
            )
            clear_of_references = all(
                np.linalg.norm(np.array(pos) - np.array(ref[:2])) >= min_distance
                for ref in reference_positions
            )
            if clear_of_obstacles and clear_of_references:
                return [pos[0], pos[1], 0.1]

    def _spawn_in_front_of_chaser(self, chaser_pos, chaser_orn):
        """Sample a runner spawn in the chaser camera's forward half-plane."""
        yaw = self._yaw_from_orientation(chaser_orn)
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        lateral = np.array([-forward[1], forward[0]], dtype=np.float32)

        distance = np.random.uniform(4.0, 8.0)
        side_offset = np.random.uniform(-2.5, 2.5)
        chaser_xy = np.asarray(chaser_pos[:2], dtype=np.float32)
        runner_xy = chaser_xy + forward * distance + lateral * side_offset

        return [float(runner_xy[0]), float(runner_xy[1]), 0.1]

    def _get_front_runner_spawn(self, chaser_pos, chaser_orn):
        for _ in range(50):
            candidate = self._spawn_in_front_of_chaser(chaser_pos, chaser_orn)

            if self._is_valid_spawn(candidate, [chaser_pos], 5.0):
                return candidate

        return self._get_valid_spawn(reference_positions=[chaser_pos], min_distance=5.0)

    def _is_valid_spawn(self, pos, reference_positions=None, min_distance=0.0):
        """Checks whether a proposed spawn is inside the arena and clear."""
        reference_positions = reference_positions or []
        xy = np.asarray(pos[:2], dtype=np.float32)
        inner_boundary = self.boundary - 1.0

        if np.any(xy < -inner_boundary) or np.any(xy > inner_boundary):
            return False

        clear_of_obstacles = all(
            np.linalg.norm(xy - np.asarray(obs_pos, dtype=np.float32)) > 1.2
            for obs_pos in self.obstacle_positions
        )
        if not clear_of_obstacles:
            return False

        return all(
            np.linalg.norm(xy - np.asarray(ref[:2], dtype=np.float32)) >= min_distance
            for ref in reference_positions
        )

    def _configure_husky_dynamics(self, robot_id):
        p.changeDynamics(
            robot_id,
            -1,
            lateralFriction=1.5,
            rollingFriction=0.005,
            spinningFriction=0.01,
            linearDamping=0.08,
            angularDamping=0.35,
            restitution=0.0,
        )

        for link_idx in range(p.getNumJoints(robot_id)):
            is_wheel = link_idx in self.wheel_joint_indices
            p.changeDynamics(
                robot_id,
                link_idx,
                lateralFriction=1.8 if is_wheel else 1.2,
                rollingFriction=0.002 if is_wheel else 0.01,
                spinningFriction=0.01,
                linearDamping=0.02,
                angularDamping=0.18,
                restitution=0.0,
            )

    # This function converts the desired linear and angular speeds into individual wheel velocities,
    # After obtaining its current position and orientation, it calculates the relative position of the runner and obstacles,
    # this way the chaser can navigate towards the runner while avoiding obstacles. 
    # It also includes debug visualization of the observation space in the GUI mode.
    # Furthermore, it gives the runner basic evasive behavior by adjusting its speed and turn rate based on the chaser's position,
    # as well simple reflexes to avoid collisions with obstacles
    def _scripted_runner_command(self):
        """Pick a runner command that escapes the chaser without driving into hazards."""
        chaser_pos, _ = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, runner_orn = p.getBasePositionAndOrientation(self.runner)

        chaser_xy = np.array(chaser_pos[:2], dtype=np.float32)
        runner_xy = np.array(runner_pos[:2], dtype=np.float32)
        runner_yaw = self._yaw_from_orientation(runner_orn)
        heading_vec = np.array([np.cos(runner_yaw), np.sin(runner_yaw)], dtype=np.float32)

        escape_vec = runner_xy - chaser_xy
        escape_dist = max(float(np.linalg.norm(escape_vec)), 1e-6)
        escape_dir = escape_vec / escape_dist

        # Try the direct escape direction plus headings around the compass.
        # The best-scoring heading becomes the runner's temporary target.
        candidate_dirs = [escape_dir]
        for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            candidate_dirs.append(np.array([np.cos(angle), np.sin(angle)], dtype=np.float32))

        lookahead_distance = 2.0
        best_score = -np.inf
        desired_vec = escape_dir
        nearest_clearance = self._runner_clearance(runner_xy)

        for direction in candidate_dirs:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue

            direction = direction / norm
            candidate_pos = runner_xy + direction * lookahead_distance
            candidate_clearance = self._runner_clearance(candidate_pos)
            candidate_chaser_distance = float(np.linalg.norm(candidate_pos - chaser_xy))

            score = 2.4 * candidate_chaser_distance
            score += 1.8 * float(np.dot(direction, escape_dir))
            score += 0.35 * float(np.dot(direction, heading_vec))
            score += 1.2 * candidate_clearance

            # Heavily reject headings that would pin the runner into walls,
            # trees, or benches, but still allow tight dodges if needed.
            if candidate_clearance < 1.0:
                score -= (1.0 - candidate_clearance) * 12.0
            if candidate_clearance < 2.5:
                score -= (2.5 - candidate_clearance) * 2.5

            # When the chaser is close, favor a lateral dodge instead of only
            # driving directly away and getting pinned into a wall.
            if escape_dist < 4.0:
                lateral_escape = abs(float(escape_dir[0] * direction[1] - escape_dir[1] * direction[0]))
                score += lateral_escape * 1.4

            # this scoring system is designed to encourage the runner to pick escape routes that are far from the chaser
            if score > best_score:
                best_score = score
                desired_vec = direction
                nearest_clearance = candidate_clearance

        # Convert the chosen escape direction into a forward speed and turn rate.
        # The runner slows down when the chaser is very close to allow for sharper turns, 
        # and speeds up when the chaser is far away for a more dynamic chase.
        desired_yaw = float(np.arctan2(desired_vec[1], desired_vec[0]))
        yaw_error = self._wrap_angle(desired_yaw - runner_yaw)
        turn_speed = float(np.clip(yaw_error * self.runner_turn_gain, -self.runner_max_turn_speed, self.runner_max_turn_speed))

        alignment = max(0.0, 1.0 - abs(yaw_error) / np.pi)
        clearance_scale = float(np.clip(nearest_clearance / 2.0, 0.45, 1.0))
        chase_pressure = float(np.clip((6.0 - escape_dist) / 6.0, 0.0, 1.0))
        forward_speed = self.runner_min_forward_speed + (
            self.runner_max_forward_speed - self.runner_min_forward_speed
        ) * (0.3 + 0.7 * alignment)
        forward_speed *= clearance_scale
        forward_speed += chase_pressure * 2.0

        if abs(yaw_error) > 1.4:
            forward_speed *= 0.55

        return forward_speed, turn_speed

    # this function calculates the relative positions of the runner and obstacles to the chaser, and also includes debug visualization of these positions in the GUI mode.
    def _runner_clearance(self, xy):
        """Distance from a candidate runner position to the nearest hazard."""
        wall_clearance = self._wall_clearance(xy)
        obstacle_clearance = self.boundary

        for obs_pos in self.obstacle_positions:
            obs_xy = np.array(obs_pos, dtype=np.float32)
            obstacle_clearance = min(obstacle_clearance, float(np.linalg.norm(xy - obs_xy)))

        return min(wall_clearance, obstacle_clearance)

    # this function calculates the distance from a candidate runner position to the nearest wall, which is used in the runner's escape behavior to avoid getting pinned against walls.
    def _wall_clearance(self, xy):
        inner_boundary = self.boundary - 0.9
        left_clearance = float(xy[0] + inner_boundary)
        right_clearance = float(inner_boundary - xy[0])
        bottom_clearance = float(xy[1] + inner_boundary)
        top_clearance = float(inner_boundary - xy[1])
        return min(left_clearance, right_clearance, bottom_clearance, top_clearance)

    # this function calculates the relative positions of the runner and obstacles to the chaser, and also includes debug visualization of these positions in the GUI mode.
    def _yaw_from_orientation(self, orientation):
        return p.getEulerFromQuaternion(orientation)[2]

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # This is an important function that blends the PPO action with a short-range obstacle avoidance reflex.
    # It checks for obstacles in front of the chaser and adjusts the action to steer around them, while still trying to pursue the runner.
    # while this isn't perfect and can sometimes cause the chaser to take a less direct path, it helps prevent the chaser from getting stuck on obstacles 
    # and allows for more dynamic pursuit behavior.
    # additionally, because the environment is constantly loaded in random configurations of trees and benches, 
    # this reflex helps the chaser learn to navigate around them effectively during training, 
    # rather than just learning to drive straight towards the runner.
    def _apply_chaser_obstacle_reflex(self, action):
        """Blend PPO action with a short-range obstacle avoidance reflex."""
        front_obstacle = self._nearest_front_obstacle()
        self.last_avoidance_turn = 0.0
        self.last_target_assist_strength = 0.0
        self.last_front_obstacle_distance = None
 
        if self.escape_steps > 0:
            self.escape_steps -= 1
            action[0] = -1.0
            action[1] = self.escape_turn_direction
            self.last_avoidance_turn = self.escape_turn_direction
            return action
 
        if front_obstacle is None:
            self.avoid_commit_steps = 0
            return self._apply_close_range_target_assist(action)

        obs_x, obs_y, obs_dist = front_obstacle
        self.last_front_obstacle_distance = obs_dist

        runner_x, runner_y = self._relative_runner_position()
        if abs(obs_y) < 0.15:
            # If the obstacle is centered, bias around it on the runner side.
            desired_turn = 1.0 if runner_y >= 0.0 else -1.0
        else:
            # Positive relative y means obstacle is left of the chaser, so turn right.
            desired_turn = -float(np.sign(obs_y))
 
        if self.avoid_commit_steps > 0:
            desired_turn = self.avoid_turn_direction
            self.avoid_commit_steps -= 1
        else:
            self.avoid_turn_direction = desired_turn
            self.avoid_commit_steps = 22
 
        if self.obstacle_contact_steps > 2 or self.stuck_steps > 12:
            self.escape_steps = 18
            self.escape_turn_direction = desired_turn
            action[0] = -1.0
            action[1] = desired_turn
            self.last_avoidance_turn = desired_turn
            return action

        # The closer the obstacle, the stronger the avoidance reflex, with a smooth falloff based on distance and lateral position.
        lookahead = 5.0
        corridor_half_width = 1.9
        forward_pressure = float(np.clip((lookahead - obs_x) / lookahead, 0.0, 1.0))
        center_pressure = float(np.clip((corridor_half_width - abs(obs_y)) / corridor_half_width, 0.0, 1.0))
        strength = forward_pressure * center_pressure
 
        turn_command = desired_turn * (0.45 + 0.45 * strength)
        action[1] = float(np.clip(action[1] * 0.25 + turn_command * 0.75, -1.0, 1.0))
 
        # Avoidance should be a forward arc. Only reverse if the obstacle is
        # directly in front and already inside the emergency buffer.
        min_forward_action = 0.15
        max_forward_action = 0.6 - 0.35 * strength
        if obs_x < 1.3 and abs(obs_y) < 0.75:
            min_forward_action = -0.35
            max_forward_action = -0.05
 
        action[0] = float(np.clip(action[0], min_forward_action, max_forward_action))
        self.last_avoidance_turn = turn_command
        return action
 
    # This function provides a subtle assist to the chaser when the runner is very close, by nudging the action towards a direct capture attempt.
    # This helps prevent the chaser from getting stuck in a close-range pursuit where it keeps circling around the runner without actually catching it,
    # which can be a common failure mode in pursuit environments.
    def _apply_close_range_target_assist(self, action):
        """Nudge clear close-range pursuits into a direct capture attempt."""
        runner_x, runner_y = self._relative_runner_position()
        runner_distance = float(np.hypot(runner_x, runner_y))
        if runner_distance >= self.near_distance:
            return action
 
        target_angle = float(np.arctan2(runner_y, max(runner_x, 1e-6)))
        turn_nudge = float(np.clip(target_angle / 1.2, -1.0, 1.0))
        assist_strength = float(
            np.clip(
                (self.near_distance - runner_distance) / max(self.near_distance - self.catch_distance, 1e-6),
                0.0,
                1.0,
            )
        )
 
        if runner_x > 0.0:
            min_forward = 0.45 + 0.55 * assist_strength
            action[0] = float(max(action[0], min_forward))
        else:
            action[0] = float(min(action[0], -0.15))
 
        action[1] = float(np.clip(action[1] * 0.55 + turn_nudge * 0.45, -1.0, 1.0))
        self.last_target_assist_strength = assist_strength
        return action
 
    # This function checks for obstacles in front of the chaser within a certain range and corridor width, and returns the relative position of the closest one.
    def _nearest_front_obstacle(self):
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        closest = None
        closest_dist = np.inf
 
        for obs_pos in self.obstacle_positions:
            rel_x, rel_y = get_relative_position(
                chaser_pos,
                chaser_orn,
                [obs_pos[0], obs_pos[1], 0.0],
            )
            if 0.0 < rel_x < 5.0 and abs(rel_y) < 1.9:
                dist = float(np.hypot(rel_x, rel_y))
                if dist < closest_dist:
                    closest = (float(rel_x), float(rel_y), dist)
                    closest_dist = dist

        return closest

    # This function calculates the relative position of the runner to the chaser, which is used in both the observation space and the runner's escape behavior.
    def _relative_runner_position(self):
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        return get_relative_position(chaser_pos, chaser_orn, runner_pos)

    # This function applies extra forces and torques to the robot to help stabilize it at high speeds and prevent it from flipping over or getting stuck on obstacles.
    def _apply_ground_stabilization(self, robot_id):
        """Apply extra damping/downforce so faster wheel speeds stay drivable."""
        base_pos, orn = p.getBasePositionAndOrientation(robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        horizontal_speed = float(np.linalg.norm(lin_vel[:2]))
        mass = p.getDynamicsInfo(robot_id, -1)[0]

        downforce = mass * (8.0 + horizontal_speed * self.downforce_scale)
        p.applyExternalForce(
            robot_id,
            -1,
            [0.0, 0.0, -downforce],
            base_pos,
            p.WORLD_FRAME,
        )

        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]], dtype=np.float32)
        right = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]], dtype=np.float32)

        if abs(pitch) > 0.05:
            # Push down on the lifted end of the chassis to counter wheelies.
            pitch_point = np.array(base_pos, dtype=np.float32) + forward * np.sign(pitch) * 0.55
            pitch_force = mass * (18.0 * abs(pitch) + horizontal_speed * 3.0)
            p.applyExternalForce(
                robot_id,
                -1,
                [0.0, 0.0, -pitch_force],
                pitch_point.tolist(),
                p.WORLD_FRAME,
            )

        if abs(roll) > 0.05:
            side_point = np.array(base_pos, dtype=np.float32) + right * np.sign(roll) * 0.35
            side_force = mass * 10.0 * abs(roll)
            p.applyExternalForce(
                robot_id,
                -1,
                [0.0, 0.0, -side_force],
                side_point.tolist(),
                p.WORLD_FRAME,
            )

        stabilizing_torque = [
            -roll * self.pitch_roll_damping - ang_vel[0] * 8.0,
            -pitch * self.pitch_roll_damping - ang_vel[1] * 8.0,
            -ang_vel[2] * 0.25,
        ]
        p.applyExternalTorque(robot_id, -1, stabilizing_torque, p.WORLD_FRAME)
            
    # This function converts the desired linear and angular speeds into individual wheel velocities,
    # and applies them to the robot's wheel joints. It also includes a ramping mechanism to prevent sudden changes in wheel speed, 
    # which helps maintain stability and control, especially when the chaser is tilted or navigating rough terrain.
    def _drive_husky(self, robot_id, lin_v, ang_v):
        # If the body is already tilted, back off the command before it flips.
        _, orn = p.getBasePositionAndOrientation(robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        tilt = max(abs(roll), abs(pitch))
        if tilt > 0.18:
            stability_scale = float(np.clip(1.0 - (tilt - 0.18) / 0.35, 0.25, 1.0))
            lin_v *= stability_scale
            ang_v *= 0.75 * stability_scale

        left, right = lin_v - ang_v, lin_v + ang_v
        left = float(np.clip(left, -self.max_wheel_velocity, self.max_wheel_velocity))
        right = float(np.clip(right, -self.max_wheel_velocity, self.max_wheel_velocity))

        current_left, current_right = self.current_wheel_targets.get(robot_id, (0.0, 0.0))
        # Ramp wheel targets instead of jumping instantly to full speed.
        left = self._ramp_value(current_left, left, self.max_wheel_accel)
        right = self._ramp_value(current_right, right, self.max_wheel_accel)
        self.current_wheel_targets[robot_id] = (left, right)

        for idx in self.wheel_joint_indices:
            p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, 
                                    targetVelocity=(left if idx % 2 == 0 else right),
                                    force=self.motor_force)

    def _ramp_value(self, current, target, max_delta):
        return current + float(np.clip(target - current, -max_delta, max_delta))

    def _has_contact_with_any(self, robot_id, body_ids):
        return any(p.getContactPoints(robot_id, body_id) for body_id in body_ids)

    # This is the core reward function that evaluates the current state of the environment 
    # and calculates the reward for the chaser's action.
    # It considers multiple factors such as distance to the runner, 
    # progress towards the runner, obstacle avoidance, and the assistive nudges applied to the action.
    # The reward is designed to encourage the chaser to pursue the runner effectively while navigating around obstacles and avoiding getting stuck,
    # rather than just rewarding proximity to the runner, which can lead to local minima where the chaser gets close but fails to actually catch the runner.
    def _calculate_reward(self):
        # contacts = p.getContactPoints(self.chaser, self.runner)
        # if contacts:
        #     return 100.0, True
        # return -0.1, False
        # Get positions

        # After obtaining the current positions and orientations of the chaser and runner, it calculates the distance between them,
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        chaser_roll, chaser_pitch, _ = p.getEulerFromQuaternion(chaser_orn)
        
        distance = float(np.linalg.norm(np.array(chaser_pos[:2]) - np.array(runner_pos[:2])))

        if self.prev_distance is None:
            progress = 0.0
        else:
            progress = self.prev_distance - distance
        self.prev_distance = distance

        # It also calculates the movement of the chaser since the last step, 
        # which can be used to reward forward progress and penalize getting stuck.
        chaser_xy = np.array(chaser_pos[:2], dtype=np.float32)
        if self.prev_chaser_xy is None:
            chaser_movement = 0.0
        else:
            chaser_movement = float(np.linalg.norm(chaser_xy - self.prev_chaser_xy))
        self.prev_chaser_xy = chaser_xy
        min_obstacle_distance = self._nearest_obstacle_distance(chaser_xy)
        wall_clearance = self._wall_clearance(chaser_xy)
        # info dictionary includes various metrics about the current state of the environment, 
        # which can be useful for debugging, analysis, and potentially for training auxiliary tasks 
        # or shaping the reward function in more complex ways.
        info = {
            "distance_to_runner": distance,
            "progress_to_runner": float(progress),
            "chaser_movement": chaser_movement,
            "min_obstacle_distance": min_obstacle_distance,
            "front_obstacle_distance": float(self.last_front_obstacle_distance or 0.0),
            "wall_clearance": wall_clearance,
            "near_target": float(distance < self.near_distance),
            "capture_pressure": float(np.clip((self.near_distance - distance) / max(self.near_distance - self.catch_distance, 1e-6), 0.0, 1.0)),
            "avoidance_reflex": float(self.last_front_obstacle_distance is not None or self.escape_steps > 0),
            "avoidance_turn": abs(float(self.last_avoidance_turn)),
            "target_assist": float(self.last_target_assist_strength),
            "raw_forward_action": float(self.last_raw_action[0]),
            "final_forward_action": float(self.last_action[0]),
            "raw_turn_action": float(self.last_raw_action[1]),
            "final_turn_action": float(self.last_action[1]),
            "stuck_steps": float(self.stuck_steps),
            "obstacle_contact_steps": float(self.obstacle_contact_steps),
            "collision": 0.0,
            "caught": 0.0,
            "is_success": 0.0,
            "terminated": 0.0,
            "fell_over": 0.0,
            "ended_stuck": 0.0,
        }
 
        # The finish function is a helper that finalizes the reward and termination status when the episode ends, 
        # whether due to catching the runner, colliding with an obstacle, getting stuck, or falling over. 
        # It updates the info dictionary with the reason for termination and whether it was a successful catch 
        # or a failure mode.
        def finish(final_reward, terminated, reason=None):
            info["terminated"] = float(terminated)
            if reason == "caught":
                info["caught"] = 1.0
                info["is_success"] = 1.0
            elif reason == "collision":
                info["collision"] = 1.0
            elif reason == "fell_over":
                info["fell_over"] = 1.0
            elif reason == "stuck":
                info["ended_stuck"] = 1.0
            return final_reward, terminated, info
        
        # Old distance-only reward kept for reference.
        # reward = -0.05
        # reward += -distance * 0.8

        # if the runner is in front of the chaser, reward more for closing the distance, and if it's behind, penalize more for not catching up.
        runner_forward_alignment = 0.0
        runner_lateral_error = 0.0
        if distance > 1e-6:
            runner_rel_x, runner_rel_y = get_relative_position(chaser_pos, chaser_orn, runner_pos)
            runner_forward_alignment = runner_rel_x / distance
            runner_lateral_error = abs(runner_rel_y) / distance

        forward_action = max(float(self.last_action[0]), 0.0)

        # Reward progress toward the runner instead of heavily punishing distance every step.
        # does this by rewarding the change in distance (progress) rather than the absolute distance, 
        # which encourages the chaser to keep moving towards the runner even if it's still far away, 
        # and also includes a small time penalty to encourage faster catches.
        reward = -0.01
        reward -= 0.015 * (self.step_count / self.max_steps)
        reward += progress * 60.0
        reward += -max(-progress, 0.0) * 12.0
        reward += -distance * 0.01
        # Small nudge to keep the chaser moving forward rather than spinning in place.
        reward += forward_action * 0.04
        close_range = max(self.near_distance - distance, 0.0)
        reward += close_range * 1.8
        reward += max(runner_forward_alignment, 0.0) * 0.35
        reward += forward_action * max(runner_forward_alignment, 0.0) * 0.12

        # if the runner is within the near distance and generally in front of the chaser, 
        # provide a stronger reward signal that encourages closing in for the capture.
        # This includes a significant reward for progress, a bonus for being aligned with the runner, 
        # and an additional incentive based on the forward action and capture pressure,
        if distance < self.near_distance and runner_forward_alignment > 0.0:
            centered_runner = max(1.0 - runner_lateral_error, 0.0)
            reward += centered_runner * close_range * 0.9
            reward += max(progress, 0.0) * 45.0
            reward -= max(-progress, 0.0) * 10.0
            capture_drive = info["capture_pressure"]
            reward += forward_action * centered_runner * (0.5 + 2.0 * capture_drive)
            if self.last_front_obstacle_distance is None:
                reward -= max(0.45 - forward_action, 0.0) * 0.8 * capture_drive
            else:
                reward += max(progress, 0.0) * 20.0
 
        if distance < self.catch_distance + 1.0:
            capture_pressure = self.catch_distance + 1.0 - distance
            reward += capture_pressure * 12.0

        # Teach PPO that catching while flat is better than exploiting unstable tilts.
        reward -= (abs(chaser_roll) + abs(chaser_pitch)) * 0.6

        # If there's an obstacle in front, reward actions that steer around it, but only if the chaser isn't already past it.
        if self.last_front_obstacle_distance is not None:
            avoid_alignment = float(self.last_raw_action[1] * self.last_avoidance_turn)
            reward += max(avoid_alignment, 0.0) * 0.35
            reward -= max(-avoid_alignment, 0.0) * 0.7
            reward += max(progress, 0.0) * 15.0
        
        # Big bonus for catching
        contacts = p.getContactPoints(self.chaser, self.runner)
        if contacts or distance <= self.catch_distance:
            return finish(800.0, True, "caught")

        # however if the chaser collides with an obstacle or gets stuck for too long, 
        # it receives a significant penalty and the episode ends,
        hit_obstacle = self._has_contact_with_any(self.chaser, self.obstacle_body_ids)
        hit_wall = self._has_contact_with_any(self.chaser, self.wall_body_ids)

        # if the chaser is in contact with an obstacle, 
        # it receives a penalty that increases with the number of consecutive steps in contact,
        # as well as a penalty for hitting the wall, which encourages the chaser to learn to avoid obstacles
        # and navigate around them rather than just crashing into them.
        # this is important for learning effective pursuit behavior in an environment with obstacles,
        # and ensure PPO is not purely reliant on the reflex to avoid obstacles, but also learns to steer around them proactively.
        if hit_obstacle:
            info["collision"] = 1.0
            reward -= 3.0 + min(self.obstacle_contact_steps, 10) * 0.15
            self.obstacle_contact_steps += 1
        elif hit_wall:
            info["collision"] = 1.0
            reward -= 4.0
            self.obstacle_contact_steps += 1
        else:
            self.obstacle_contact_steps = max(0, self.obstacle_contact_steps - 1)

        # If the chaser is trying to move but barely changes position, it is
        # probably pinned against an obstacle. Resetting this state helps PPO
        # learn to avoid the behavior instead of sitting there until timeout.
        if forward_action > 0.4 and chaser_movement < 0.01 and distance > self.catch_distance:
            self.stuck_steps += 1
            reward -= min(self.stuck_steps, 50) * 0.05
        else:
            self.stuck_steps = max(0, self.stuck_steps - 2)

        if self.obstacle_contact_steps >= 45:
            return finish(-35.0, True, "collision")
 
        if self.stuck_steps >= 80:
            return finish(-35.0, True, "stuck")
        
        # Obstacle avoidance penalty — scaled down when the runner is beyond the
        # obstacle (the chaser MUST go near it to continue pursuit).
        runner_pos_arr = np.array(runner_pos[:2], dtype=np.float32)
        for obs_pos in self.obstacle_positions:
            obs_arr = np.array(obs_pos, dtype=np.float32)
            obs_dist = np.linalg.norm(chaser_xy - obs_arr)
            if obs_dist < 2.0:
                # Reduce penalty if the runner is on the far side of this obstacle
                # so the chaser isn't punished for navigating a necessary gap.
                runner_beyond = float(np.linalg.norm(runner_pos_arr - obs_arr)) < obs_dist
                scale = 0.3 if runner_beyond else 1.1
                reward -= (2.0 - obs_dist) * scale
 
            # get the relative position of the obstacle to the chaser, 
            # and if it's within a certain range in front of the chaser, 
            # apply a penalty that encourages the chaser to steer around it 
            # rather than crashing into it.
            obs_rel_x, obs_rel_y = get_relative_position(chaser_pos, chaser_orn, [obs_pos[0], obs_pos[1], 0.0])
            if 0.0 < obs_rel_x < 3.6 and abs(obs_rel_y) < 1.35:
                reward -= (3.6 - obs_rel_x) * 0.24
        
        # Wall penalty
        if abs(chaser_pos[0]) > self.boundary - 1.0 or abs(chaser_pos[1]) > self.boundary - 1.0:
            reward -= 5.0

        # If the chaser flips over, end the episode with a penalty. 
        # This encourages the chaser to learn to maintain stability 
        # while pursuing the runner, rather than just going all out 
        # and risking a flip.
        if abs(chaser_roll) > 1.0 or abs(chaser_pitch) > 1.0:
            return finish(-100.0, True, "fell_over")
        
        terminated = False
        return finish(reward, terminated)
 
    # This function calculates the distance from the chaser to the 
    # nearest obstacle, which is used in the reward function to 
    # encourage the chaser to maintain a safe distance from obstacles 
    # while pursuing the runner.
    def _nearest_obstacle_distance(self, xy):
        if not self.obstacle_positions:
            return float(self.boundary)
 
        return float(
            min(
                np.linalg.norm(np.asarray(xy, dtype=np.float32) - np.asarray(obs_pos, dtype=np.float32))
                for obs_pos in self.obstacle_positions
            )
        )

    # This function resets the environment to a new random configuration, 
    # including the positions of the chaser, runner, and obstacles.
    # It also resets all the internal state variables and returns the initial observation.
    def close(self):
        if p.isConnected():
            p.disconnect()

    # The destructor ensures that the PyBullet connection is properly closed 
    # when the environment object is deleted, which helps prevent resource 
    # leaks and ensures a clean shutdown of the simulation.
    def __del__(self):
        self.close()

# This block allows the environment to be run as a standalone script 
# for testing and debugging purposes.
if __name__ == "__main__":
    env = HuskyChaserEnv()
    try:
        obs, _ = env.reset()
        print("✅ Environment started successfully! Observation shape:", obs.shape)
        
        while True:
            action = np.array([0.5, 0.0])
            obs, reward, terminated, _, _ = env.step(action)
            time.sleep(1./240.)
            
            if terminated:
                print("Caught the runner!")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()
 
