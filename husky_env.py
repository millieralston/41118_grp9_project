import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import random

# Importing observation space functions from perception.py
from perception import create_observation_space
from perception import get_observation

class HuskyChaserEnv(gym.Env):
    def __init__(self):
        super(HuskyChaserEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.observation_space = create_observation_space()
        
        # Configuration
        self.boundary = 10  # Fence distance from center
        self.obstacle_positions = []

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Initialising IDs for observation space debug visuals
        self.runner_line_id = -1
        self.runner_text_id = -1

        self.obstacle_line_ids = []
        self.obstacle_text_ids = []

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
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    def _create_tree(self, pos):
        self.obstacle_positions.append(pos)
        trunk_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.5)
        trunk_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.5, rgbaColor=[0.5, 0.3, 0.1, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=trunk_col, baseVisualShapeIndex=trunk_vis, basePosition=[pos[0], pos[1], 0.75])
        
        leaf_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.8, rgbaColor=[0.1, 0.6, 0.2, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=leaf_vis, basePosition=[pos[0], pos[1], 1.8])

    def _create_bench(self, pos):
        self.obstacle_positions.append(pos)
        bench_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.15])
        bench_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.15], rgbaColor=[0.4, 0.3, 0.2, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bench_col, baseVisualShapeIndex=bench_vis, basePosition=[pos[0], pos[1], 0.15])

    def _get_valid_spawn(self):
        """Ensures the robot doesn't spawn on top of an obstacle."""
        while True:
            # Pick a random spot inside the fence
            pos = [random.uniform(-self.boundary + 1, self.boundary - 1), 
                   random.uniform(-self.boundary + 1, self.boundary - 1)]
            
            # Check distance against all placed obstacles (trees/benches)
            if all(np.linalg.norm(np.array(pos) - np.array(obs)) > 1.2 for obs in self.obstacle_positions):
                return [pos[0], pos[1], 0.1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.obstacle_positions = [] # Clear tracking list
        
        p.loadURDF("plane.urdf")
        self._build_fence()

        # Place Obstacles
        for _ in range(12):
            self._create_tree([random.uniform(-8, 8), random.uniform(-8, 8)])
        for _ in range(5):
            self._create_bench([random.uniform(-7, 7), random.uniform(-7, 7)])

        # Spawn Robots safely
        chaser_pos = self._get_valid_spawn()
        self.chaser = p.loadURDF("husky/husky.urdf", basePosition=chaser_pos)
        p.changeVisualShape(self.chaser, -1, rgbaColor=[1, 0, 0, 1])
        
        runner_pos = self._get_valid_spawn()
        self.runner = p.loadURDF("husky/husky.urdf", basePosition=runner_pos)
        p.changeVisualShape(self.runner, -1, rgbaColor=[0, 0, 1, 1])
        
        return self._get_obs(), {}

    def step(self, action):
        self._drive_husky(self.chaser, action[0] * 4.0, action[1] * 2.0)
        self._drive_husky(self.runner, 3.0, 0.3) # Automated runner
        p.stepSimulation()
        
        self.get_camera_image() # Display/update camera feed

        obs = self._get_obs()
        reward, terminated = self._calculate_reward()
        return obs, reward, terminated, False, {}

    # def _get_obs(self):
    #     pos, orn = p.getBasePositionAndOrientation(self.chaser)
    #     rot_matrix = p.getMatrixFromQuaternion(orn)
    #     forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
        
    #     cam_pos = [pos[0] + forward[0]*0.5, pos[1] + forward[1]*0.5, pos[2] + 0.6]
    #     target = [pos[0] + forward[0]*3, pos[1] + forward[1]*3, pos[2] + 0.4]
        
    #     view_matrix = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
    #     proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
    #     img = p.getCameraImage(64, 64, view_matrix, proj_matrix)[2]
    #     print(np.array(img).shape) # DEBUGGING
    #     return np.array(img, dtype=np.uint8)[:, :, :3]
    
    def get_camera_image(self):
        pos, orn = p.getBasePositionAndOrientation(self.chaser)

        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

        cam_pos = [
            pos[0] + forward[0] * 0.5,
            pos[1] + forward[1] * 0.5,
            pos[2] + 0.6
        ]

        target = [
            pos[0] + forward[0] * 3,
            pos[1] + forward[1] * 3,
            pos[2] + 0.4
        ]

        view_matrix = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)

        _, _, img, _, _ = p.getCameraImage(
            width=64,
            height=64,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )

        img = np.array(img, dtype=np.uint8)

        # Handle flattened image buffers
        if img.ndim == 1:
            img = img.reshape((64, 64, 4))

        # Remove alpha channel if present
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        return img
    
    def _get_obs(self):
        return get_observation(self)

    def _drive_husky(self, robot_id, lin_v, ang_v):
        left, right = lin_v - ang_v, lin_v + ang_v
        for idx in [2, 3, 4, 5]:
            p.setJointMotorControl2(robot_id, idx, p.VELOCITY_CONTROL, 
                                    targetVelocity=(left if idx % 2 == 0 else right))

    def _calculate_reward(self):
        contacts = p.getContactPoints(self.chaser, self.runner)
        if contacts: return 100.0, True
        return -0.1, False

if __name__ == "__main__":
    env = HuskyChaserEnv()
    env.reset()
    try:
        while True:
            env.step(np.array([0.7, 0.2]))
            time.sleep(1./60.)
    except KeyboardInterrupt:
        p.disconnect()