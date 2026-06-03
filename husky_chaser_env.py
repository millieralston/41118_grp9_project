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

# PyBullet prints repeated URDF warnings when loading the Husky model. They are
# not PPO training errors, so this keeps console/TensorBoard output readable
# while still allowing our own logs and exceptions to show normally.
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
        # PPO controls only the red chaser. The two action values are normalized
        # to [-1, 1] and later mapped to forward drive and turn rate.
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Old CNN/image observation space kept for reference.
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # MLP/PPO uses a compact 8-value vector from perception.py instead of
        # camera pixels: runner relative x/y plus the three closest obstacles.
        self.observation_space = create_observation_space()
        
        # boundary=10 means the fenced arena spans x/y -10 to +10, so it is a
        # 20x20 play area. The policy does not directly observe the walls, so
        # wall/obstacle clearance is handled through reset validity, runner
        # behavior, reward penalties, and info metrics.
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

        # PPO episodes last up to 1000 environment steps. That is long enough
        # for routing around trees/benches, reacquiring the runner, and making
        # a final catch instead of ending too early after one bad turn.
        self.max_steps = 1000
        self.step_count = 0

        # Movement tuning values. These are PyBullet wheel target velocities,
        # not real m/s. PPO outputs normalized actions, then step() maps them
        # into this range. The chaser can be faster than the scripted runner
        # (22.5 vs 13.0 forward target speed), but acceleration, max wheel speed,
        # downforce, and damping stop the learned policy from exploiting flips.
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

    # reset() starts a fresh PPO episode. It rebuilds the PyBullet world,
    # randomizes 12 trees and 5 benches, spawns the chaser clear of obstacles,
    # then spawns the runner either randomly at least 5 units away or in front
    # for demos. It also resets all reward-memory variables such as previous
    # distance, stuck counters, and reflex state before returning the first
    # observation vector.
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

        # Reset runner visibility/cache used by the optional YOLO demo path.
        # PPO training still uses the relative-position vector observation.
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

    # Obtain the runner and nearby obstacle positions relative to the chaser.
    # This vector is what PPO sees at every step.
    def _get_obs(self):
        # Old CNN/image observation kept for reference.
        # return self.get_camera_image()

        return get_observation(self)

    # Camera helpers are used for previews, YOLO demos, and old CNN experiments.
    # The current PPO policy does not train from these images.
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

    # step() is the PPO transition function. It takes the policy action,
    # applies small safety/capture helpers, drives both Huskies through several
    # physics substeps, then returns (observation, shaped reward, done flags,
    # info metrics) for the next PPO update.
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

    # Shared random spawn helper. It samples inside the fence with a 1.0 unit
    # margin, rejects positions within 1.2 units of trees/benches, and can also
    # enforce a minimum distance from another robot such as the runner/chaser.
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

        # Demo/front mode puts the runner 4-8 units ahead with a small lateral
        # offset, then _get_front_runner_spawn() validates that it is still
        # inside the arena and clear of obstacles.
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

    # Scripted runner used as the moving target for PPO. It is not learned:
    # each step it scores escape headings by distance from the chaser, alignment
    # away from the chaser, current heading, and clearance from walls/obstacles.
    # This gives PPO a consistent but non-trivial opponent, so the learned
    # chaser must pursue, turn, and recover rather than just drive at a static
    # target.
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

    # Clearance is used by the scripted runner to avoid escape routes that would
    # trap it against walls, trees, or benches. This keeps episodes useful for
    # PPO because the runner stays mobile instead of getting stuck immediately.
    def _runner_clearance(self, xy):
        """Distance from a candidate runner position to the nearest hazard."""
        wall_clearance = self._wall_clearance(xy)
        obstacle_clearance = self.boundary

        for obs_pos in self.obstacle_positions:
            obs_xy = np.array(obs_pos, dtype=np.float32)
            obstacle_clearance = min(obstacle_clearance, float(np.linalg.norm(xy - obs_xy)))

        return min(wall_clearance, obstacle_clearance)

    # Distance from a candidate x/y position to the nearest wall. With
    # boundary=10 and a 0.9 unit margin, values near 0 mean the robot is almost
    # touching the fence.
    def _wall_clearance(self, xy):
        inner_boundary = self.boundary - 0.9
        left_clearance = float(xy[0] + inner_boundary)
        right_clearance = float(inner_boundary - xy[0])
        bottom_clearance = float(xy[1] + inner_boundary)
        top_clearance = float(inner_boundary - xy[1])
        return min(left_clearance, right_clearance, bottom_clearance, top_clearance)

    def _yaw_from_orientation(self, orientation):
        return p.getEulerFromQuaternion(orientation)[2]

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # Short-range safety reflex layered on top of the PPO action. This is not
    # the main strategy: PPO still receives rewards/penalties for progress,
    # collisions, wall clearance, and stuck states. The reflex only prevents
    # common dead-end crashes by steering around obstacles detected in a
    # 5.0-unit lookahead corridor and briefly reversing if the chaser is already
    # pinned.
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

        # Reflex strength is highest for obstacles that are close and centered:
        # lookahead=5.0, corridor half-width=1.9. The turn is blended with the
        # original PPO turn, so the policy can still learn useful steering
        # instead of being completely overwritten.
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
 
    # Close-range target assist. When the runner is inside near_distance=5.0
    # and no obstacle reflex is active, the action is nudged toward the runner
    # and given minimum forward drive. The assist strength ramps to 1.0 at the
    # catch distance of 2.3, mainly preventing endless circling during capture.
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
 
    # Finds the nearest tree/bench in the chaser's forward corridor. Relative x
    # must be 0-5 units ahead and relative y within +/-1.9 units, matching the
    # reflex lookahead used above.
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

    # Relative runner position in the chaser frame. Positive x is ahead of the
    # chaser; y is lateral error. This same coordinate system is used by the
    # observation, target assist, and reward alignment terms.
    def _relative_runner_position(self):
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        return get_relative_position(chaser_pos, chaser_orn, runner_pos)

    # Stabilization keeps the high wheel speeds trainable. Extra downforce and
    # roll/pitch damping reduce wheelies, so PPO is rewarded for pursuit skill
    # rather than unstable physics exploits.
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
            
    # Convert linear/turn commands into left/right wheel velocities. Targets are
    # clipped to +/-28 and ramped by at most 3.0 per environment step to keep
    # control smooth enough for PPO to learn from.
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

    # Main reward shaping for PPO. The policy is rewarded for reducing distance
    # to the runner, staying aligned, driving forward when useful, closing the
    # final 5.0-unit capture range, and actually catching the runner. It is
    # penalized for moving away, collisions, wall/obstacle crowding, getting
    # stuck, and unstable roll/pitch. The reflex can help avoid immediate
    # crashes, but these rewards are what teach PPO the pursuit behavior.
    def _calculate_reward(self):
        # contacts = p.getContactPoints(self.chaser, self.runner)
        # if contacts:
        #     return 100.0, True
        # return -0.1, False
        # Get positions

        # Distance and progress are the primary pursuit signal. Positive
        # progress means the current step reduced the chaser-runner distance.
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        chaser_roll, chaser_pitch, _ = p.getEulerFromQuaternion(chaser_orn)
        
        distance = float(np.linalg.norm(np.array(chaser_pos[:2]) - np.array(runner_pos[:2])))

        if self.prev_distance is None:
            progress = 0.0
        else:
            progress = self.prev_distance - distance
        self.prev_distance = distance

        # Movement and clearance values are logged and also used to detect
        # failure cases such as being pinned against an obstacle.
        chaser_xy = np.array(chaser_pos[:2], dtype=np.float32)
        if self.prev_chaser_xy is None:
            chaser_movement = 0.0
        else:
            chaser_movement = float(np.linalg.norm(chaser_xy - self.prev_chaser_xy))
        self.prev_chaser_xy = chaser_xy
        min_obstacle_distance = self._nearest_obstacle_distance(chaser_xy)
        wall_clearance = self._wall_clearance(chaser_xy)
        # info exposes the important PPO diagnostics to train.py/TensorBoard:
        # distance, progress, obstacle/wall clearance, whether reflex/assist
        # modified the action, and episode-ending failure/success flags.
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
 
        # Helper for terminal rewards and TensorBoard flags.
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

        # Alignment tells the reward whether the runner is generally in front
        # of the chaser and whether the chaser is centered for a clean catch.
        runner_forward_alignment = 0.0
        runner_lateral_error = 0.0
        if distance > 1e-6:
            runner_rel_x, runner_rel_y = get_relative_position(chaser_pos, chaser_orn, runner_pos)
            runner_forward_alignment = runner_rel_x / distance
            runner_lateral_error = abs(runner_rel_y) / distance

        forward_action = max(float(self.last_action[0]), 0.0)

        # Base reward numbers for the video explanation:
        # -0.01 step cost, an extra time cost up to -0.015 by episode end,
        # +60 * progress for closing distance, -12 * lost progress when moving
        # away, and only -0.01 * distance so the policy is not dominated by
        # being far away early in the episode.
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

        # Inside near_distance=5.0 the reward becomes more capture-focused:
        # centered close range adds up to +0.9 * close_range, progress gets an
        # extra +45 multiplier, moving away gets -10, and forward drive is
        # rewarded more as capture_pressure approaches 1.0.
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
        
        # A catch is either physical contact or distance <= catch_distance=2.3.
        # It ends the episode with +800, much larger than shaping rewards, so
        # PPO learns that the final objective is capture rather than hovering.
        contacts = p.getContactPoints(self.chaser, self.runner)
        if contacts or distance <= self.catch_distance:
            return finish(800.0, True, "caught")

        hit_obstacle = self._has_contact_with_any(self.chaser, self.obstacle_body_ids)
        hit_wall = self._has_contact_with_any(self.chaser, self.wall_body_ids)

        # Collision shaping makes PPO learn proactive avoidance instead of
        # relying only on the reflex. Obstacle contact starts at -3.0 and grows
        # by 0.15 per consecutive contact step up to 10 steps; wall contact is
        # -4.0. Long contact still ends the episode, so crashing is not a valid
        # shortcut.
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
        # probably pinned against an obstacle. The penalty grows by -0.05 per
        # stuck step up to 50 steps, and 80 stuck steps ends the episode.
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
 
            # Extra forward-corridor penalty for obstacles 0-3.6 units ahead
            # and within +/-1.35 lateral units. This teaches earlier steering,
            # not just reacting after contact.
            obs_rel_x, obs_rel_y = get_relative_position(chaser_pos, chaser_orn, [obs_pos[0], obs_pos[1], 0.0])
            if 0.0 < obs_rel_x < 3.6 and abs(obs_rel_y) < 1.35:
                reward -= (3.6 - obs_rel_x) * 0.24
        
        # Wall penalty: crossing the 1.0 unit inner margin costs -5.0.
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
 
    # Nearest obstacle distance is logged to TensorBoard and used in reward
    # shaping to measure whether the policy is learning safer paths.
    def _nearest_obstacle_distance(self, xy):
        if not self.obstacle_positions:
            return float(self.boundary)
 
        return float(
            min(
                np.linalg.norm(np.asarray(xy, dtype=np.float32) - np.asarray(obs_pos, dtype=np.float32))
                for obs_pos in self.obstacle_positions
            )
        )

    # Close the PyBullet connection when training/demo code is finished.
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
 
