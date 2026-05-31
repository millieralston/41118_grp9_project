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
from perception import create_observation_space, get_observation, get_relative_position


@contextlib.contextmanager
def suppress_pybullet_load_warnings():
    """Hide noisy URDF importer warnings while keeping normal training logs visible."""
    saved_stdout = None
    saved_stderr = None
    devnull = None

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
    def __init__(self, render_mode="direct"): # "gui" for visualization, "direct" for fast training
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Old CNN/image observation space kept for reference.
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # MLP/PPO uses a small vector from perception.py instead of camera pixels.
        self.observation_space = create_observation_space()
        
        self.boundary = 10
        self.obstacle_positions = []
        self.obstacle_body_ids = []
        self.wall_body_ids = []

        self.render_mode = render_mode.lower()
        self.debug_perception = self.render_mode == "gui"
        # Longer episodes give the chaser time to route around obstacles,
        # reacquire the runner, and finish the catch.
        self.max_steps = 800
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
        self.catch_distance = 1.8
        self.near_distance = 4.5
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_avoidance_turn = 0.0
        self.last_front_obstacle_distance = None
        self.escape_steps = 0
        self.escape_turn_direction = 1.0
        # self.runner_ang = 0.0
        # self.runner_turn_timer = 0
        
        # Initialising IDs for observation space debug visuals
        self.runner_line_id = -1
        self.runner_text_id = -1

        self.obstacle_line_ids = []
        self.obstacle_text_ids = []

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
        self.step_count = 0
        self.prev_distance = None
        self.prev_chaser_xy = None
        self.stuck_steps = 0
        self.obstacle_contact_steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_raw_action = np.zeros(2, dtype=np.float32)
        self.last_avoidance_turn = 0.0
        self.last_front_obstacle_distance = None
        self.escape_steps = 0
        self.escape_turn_direction = 1.0
        self.current_wheel_targets = {}
        
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
        
        # runner_pos = self._get_valid_spawn(reference_positions=[chaser_pos], min_distance=5.0)
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)

        runner_pos = None

        for _ in range(50):
            candidate = self._spawn_in_front_of_chaser(chaser_pos, chaser_orn)

            if self._is_valid_spawn(candidate, [chaser_pos], 2.0):
                runner_pos = candidate
                break

        if runner_pos is None:
            for _ in range(20):
                candidate = chaser_pos + np.array([np.random.uniform(1.5, 3.0), np.random.uniform(-2, 2), 0])

                if self._is_valid_spawn(candidate, [chaser_pos], 2.0):
                    runner_pos = candidate
                    break
            else:
                runner_pos = chaser_pos  # last-resort safe state

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

    def get_camera_image(self):
        pos, orn = p.getBasePositionAndOrientation(self.chaser)
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        
        cam_pos = [pos[0] + forward[0]*0.5, pos[1] + forward[1]*0.5, pos[2] + 0.6]
        target = [pos[0] + forward[0]*3, pos[1] + forward[1]*3, pos[2] + 0.4]
        
        view_matrix = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
        
        img_data = p.getCameraImage(64, 64, view_matrix, proj_matrix, 
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb = img_data[2]
        if isinstance(rgb, tuple):
            rgb = np.array(rgb, dtype=np.uint8).reshape(64, 64, 4)
        else:
            rgb = np.array(rgb, dtype=np.uint8).reshape(64, 64, 4)
        
        return rgb[:, :, :3]

    # step function applies the action to the chaser, moves the runner, steps the simulation, and calculates the reward and termination status.
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
        reward, terminated = self._calculate_reward()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

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

    def _runner_clearance(self, xy):
        """Distance from a candidate runner position to the nearest hazard."""
        wall_clearance = self._wall_clearance(xy)
        obstacle_clearance = self.boundary

        for obs_pos in self.obstacle_positions:
            obs_xy = np.array(obs_pos, dtype=np.float32)
            obstacle_clearance = min(obstacle_clearance, float(np.linalg.norm(xy - obs_xy)))

        return min(wall_clearance, obstacle_clearance)

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

    def _apply_chaser_obstacle_reflex(self, action):
        """Blend PPO action with a short-range obstacle avoidance reflex."""
        front_obstacle = self._nearest_front_obstacle()
        self.last_avoidance_turn = 0.0
        self.last_front_obstacle_distance = None

        if self.escape_steps > 0:
            self.escape_steps -= 1
            action[0] = -1.0
            action[1] = self.escape_turn_direction
            self.last_avoidance_turn = self.escape_turn_direction
            return action

        if front_obstacle is None:
            return action

        obs_x, obs_y, obs_dist = front_obstacle
        self.last_front_obstacle_distance = obs_dist

        runner_x, runner_y = self._relative_runner_position()
        if abs(obs_y) < 0.15:
            # If the obstacle is centered, bias around it on the runner side.
            desired_turn = 1.0 if runner_y >= 0.0 else -1.0
        else:
            # Positive relative y means obstacle is left of the chaser, so turn right.
            desired_turn = -float(np.sign(obs_y))

        if self.obstacle_contact_steps > 2 or self.stuck_steps > 12:
            self.escape_steps = 18
            self.escape_turn_direction = desired_turn
            action[0] = -1.0
            action[1] = desired_turn
            self.last_avoidance_turn = desired_turn
            return action

        lookahead = 3.2
        corridor_half_width = 1.4
        forward_pressure = float(np.clip((lookahead - obs_x) / lookahead, 0.0, 1.0))
        center_pressure = float(np.clip((corridor_half_width - abs(obs_y)) / corridor_half_width, 0.0, 1.0))
        strength = forward_pressure * center_pressure

        turn_boost = desired_turn * (0.35 + 0.65 * strength)
        if obs_dist < 1.2:
            action[1] = desired_turn
        else:
            action[1] = float(np.clip(action[1] + turn_boost, -1.0, 1.0))

        # Slow into a turn instead of stopping. Stopping in front of an obstacle
        # often leaves the chaser waiting instead of driving around it.
        max_forward_action = 0.15 - 1.05 * strength
        if obs_dist < 1.2:
            max_forward_action = -0.9
        action[0] = float(min(action[0], max_forward_action))
        self.last_avoidance_turn = turn_boost
        return action

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
            if 0.0 < rel_x < 3.2 and abs(rel_y) < 1.4:
                dist = float(np.hypot(rel_x, rel_y))
                if dist < closest_dist:
                    closest = (float(rel_x), float(rel_y), dist)
                    closest_dist = dist

        return closest

    def _relative_runner_position(self):
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        return get_relative_position(chaser_pos, chaser_orn, runner_pos)

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


    def _calculate_reward(self):
        # contacts = p.getContactPoints(self.chaser, self.runner)
        # if contacts:
        #     return 100.0, True
        # return -0.1, False
        # Get positions
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self.chaser)
        runner_pos, _ = p.getBasePositionAndOrientation(self.runner)
        chaser_roll, chaser_pitch, _ = p.getEulerFromQuaternion(chaser_orn)
        
        distance = np.linalg.norm(np.array(chaser_pos[:2]) - np.array(runner_pos[:2]))

        if self.prev_distance is None:
            progress = 0.0
        else:
            progress = self.prev_distance - distance
        self.prev_distance = distance

        chaser_xy = np.array(chaser_pos[:2], dtype=np.float32)
        if self.prev_chaser_xy is None:
            chaser_movement = 0.0
        else:
            chaser_movement = float(np.linalg.norm(chaser_xy - self.prev_chaser_xy))
        self.prev_chaser_xy = chaser_xy
        
        # Old distance-only reward kept for reference.
        # reward = -0.05
        # reward += -distance * 0.8

        runner_forward_alignment = 0.0
        runner_lateral_error = 0.0
        if distance > 1e-6:
            runner_rel_x, runner_rel_y = get_relative_position(chaser_pos, chaser_orn, runner_pos)
            runner_forward_alignment = runner_rel_x / distance
            runner_lateral_error = abs(runner_rel_y) / distance

        forward_action = max(float(self.last_action[0]), 0.0)

        # Reward progress toward the runner instead of heavily punishing distance every step.
        reward = -0.01
        reward -= 0.015 * (self.step_count / self.max_steps)
        reward += progress * 50.0
        reward += -max(-progress, 0.0) * 15.0
        reward += -distance * 0.01
        close_range = max(self.near_distance - distance, 0.0)
        reward += close_range * 1.8
        reward += max(runner_forward_alignment, 0.0) * 0.35
        reward += forward_action * max(runner_forward_alignment, 0.0) * 0.12

        if distance < self.near_distance and runner_forward_alignment > 0.0:
            centered_runner = max(1.0 - runner_lateral_error, 0.0)
            reward += centered_runner * close_range * 0.6
            reward += max(progress, 0.0) * 35.0
            reward -= max(-progress, 0.0) * 8.0

        if distance < self.catch_distance + 1.0:
            capture_pressure = self.catch_distance + 1.0 - distance
            reward += capture_pressure * 6.0
        # Teach PPO that catchsiing while flat is better than exploiting unstable tilts.
        reward -= (abs(chaser_roll) + abs(chaser_pitch)) * 0.6

        if self.last_front_obstacle_distance is not None:
            avoid_alignment = float(self.last_raw_action[1] * self.last_avoidance_turn)
            reward += max(avoid_alignment, 0.0) * 0.35
            reward -= max(-avoid_alignment, 0.0) * 0.7
            reward -= max(float(self.last_raw_action[0]), 0.0) * 0.15
        
        # Big bonus for catching
        contacts = p.getContactPoints(self.chaser, self.runner)
        if contacts or distance <= self.catch_distance:
            return 800.0, True

        hit_obstacle = self._has_contact_with_any(self.chaser, self.obstacle_body_ids)
        hit_wall = self._has_contact_with_any(self.chaser, self.wall_body_ids)

        if hit_obstacle:
            reward -= 3.0 + min(self.obstacle_contact_steps, 10) * 0.15
            self.obstacle_contact_steps += 1
        elif hit_wall:
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
            return -35.0, True

        if self.stuck_steps >= 80:
            return -35.0, True
        
        # Obstacle avoidance penalty
        for obs_pos in self.obstacle_positions:
            obs_dist = np.linalg.norm(np.array(chaser_pos[:2]) - np.array(obs_pos))
            if obs_dist < 1.5:
                reward -= (1.5 - obs_dist) * 0.8   # Penalty when too close

            obs_rel_x, obs_rel_y = get_relative_position(chaser_pos, chaser_orn, [obs_pos[0], obs_pos[1], 0.0])
            if 0.0 < obs_rel_x < 2.5 and abs(obs_rel_y) < 1.1:
                reward -= (2.5 - obs_rel_x) * 0.18
        
        # Wall penalty
        if abs(chaser_pos[0]) > self.boundary - 1.0 or abs(chaser_pos[1]) > self.boundary - 1.0:
            reward -= 5.0

        if abs(chaser_roll) > 1.0 or abs(chaser_pitch) > 1.0:
            return -100.0, True
        
        terminated = False
        return reward, terminated

    def _spawn_in_front_of_chaser(self, chaser_pos, chaser_orn, lateral_spread=2.0, height=0.1):
        """
        Spawns runner in front of the chaser with some randomness.
        """

        # Get rotation matrix from quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(chaser_orn))

        # Forward and right vectors (PyBullet uses row-major 3x3 matrix)
        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        right = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])

        forward = forward.astype(np.float32)
        right   = right.astype(np.float32)

        forward = forward / (np.linalg.norm(forward) + 1e-6)
        right   = right / (np.linalg.norm(right) + 1e-6)

        # Random distance in front of chaser
        r = np.random.rand()

        if r < 0.3:
            dist = np.random.uniform(4.0, 8.0) # close
        elif r < 0.7:
            dist = np.random.uniform(8.0, 12.0) # medium
        else:
            dist = np.random.uniform(12.0, 16.0) # far

        # Random sideways offset
        side = np.random.uniform(-lateral_spread, lateral_spread)

        # Compute spawn position
        spawn = (np.array(chaser_pos, dtype=np.float32) + forward * dist + right * side)

        z = self._get_ground_height(spawn[0], spawn[1])
        spawn[2] = z + 0.1  # keep on ground plane

        return spawn.tolist()
    
    def _is_valid_spawn(self, pos, reference_positions=None, min_distance=0.0):
        reference_positions = reference_positions or []

        clear_of_obstacles = all(
            np.linalg.norm(np.array(pos[:2]) - np.array(obs)) > 1.2
            for obs in self.obstacle_positions
        )

        clear_of_references = all(
            np.linalg.norm(np.array(pos[:2]) - np.array(ref[:2])) >= min_distance
            for ref in reference_positions
        )

        inside_bounds = (
            -self.boundary + 1 < pos[0] < self.boundary - 1 and
            -self.boundary + 1 < pos[1] < self.boundary - 1
        )

        not_colliding = not self._collides_with_world(pos)

        return (
            # clear_of_obstacles and
            clear_of_references and
            inside_bounds and
            not_colliding
        )

    def _get_ground_height(self, x, y):
        ray_start = [x, y, 10]
        ray_end   = [x, y, -10]

        hit = p.rayTest(ray_start, ray_end)[0]

        return hit[3][2]  # z position of hit point

    def _collides_with_world(self, pos, radius=0.5):
        aabb_min = [pos[0] - radius, pos[1] - radius, pos[2] - 0.2]
        aabb_max = [pos[0] + radius, pos[1] + radius, pos[2] + 0.5]

        overlaps = p.getOverlappingObjects(aabb_min, aabb_max)

        if overlaps is None:
            return False

        for obj_id, _ in overlaps:
            if obj_id in self.obstacle_body_ids:  # fences, walls, trees, benches
                return True

        return False

    def close(self):
        if p.isConnected():
            p.disconnect()

    def __del__(self):
        self.close()


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
