import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import random
import math

class MouseMazeEnv(gym.Env):
    def __init__(self, renders=False, maze_size=11):
        super(MouseMazeEnv, self).__init__()
        self.renders = renders
        self.maze_size = maze_size
        self.cell_size = 1.0
        
        # Connect to PyBullet
        self.client = p.connect(p.GUI if renders else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Action space: [x_force, y_force] continuous
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: [rel_goal_x, rel_goal_y, mouse_vel_x, mouse_vel_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.mouse_id = None
        self.cheese_id = None
        self.maze = self._generate_maze(maze_size, maze_size)

    def _generate_maze(self, w, h):
        maze = [[1 for _ in range(w)] for _ in range(h)]
        def walk(x, y):
            maze[y][x] = 0
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx*2, y + dy*2
                if 0 <= nx < w and 0 <= ny < h and maze[ny][nx] == 1:
                    maze[y + dy][x + dx] = 0
                    walk(nx, ny)
        walk(1, 1)
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Build Maze
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                if self.maze[r][c] == 1:
                    p.createMultiBody(0, wall_shape, -1, [c, r, 0.5])

        # Spawn Mouse
        self.mouse_id = p.createMultiBody(1, p.createCollisionShape(p.GEOM_SPHERE, radius=0.3), 
                                         p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0,0,1,1]), 
                                         [1, 1, 0.3])
        
        # Spawn Cheese
        self.cheese_id = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.2]),
                                          p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.2], rgbaColor=[1,1,0,1]),
                                          [self.maze_size-2, self.maze_size-2, 0.2])
        
        self.prev_dist = self._get_dist()
        return self._get_obs(), {}

    def _get_dist(self):
        m_pos, _ = p.getBasePositionAndOrientation(self.mouse_id)
        c_pos, _ = p.getBasePositionAndOrientation(self.cheese_id)
        return math.sqrt((m_pos[0]-c_pos[0])**2 + (m_pos[1]-c_pos[1])**2)

    def _get_obs(self):
        m_pos, m_orn = p.getBasePositionAndOrientation(self.mouse_id)
        c_pos, c_orn = p.getBasePositionAndOrientation(self.cheese_id)
        
        # Relative coordinate transformation (as per your task standard)
        inv_m_pos, inv_m_orn = p.invertTransform(m_pos, m_orn)
        rel_goal, _ = p.multiplyTransforms(inv_m_pos, inv_m_orn, c_pos, [0,0,0,1])
        
        vel, _ = p.getBaseVelocity(self.mouse_id)
        return np.array([rel_goal[0], rel_goal[1], vel[0], vel[1]], dtype=np.float32)

    def step(self, action):
        # Apply force
        p.applyExternalForce(self.mouse_id, -1, [action[0]*10, action[1]*10, 0], [0,0,0], p.WORLD_FRAME)
        p.stepSimulation()
        
        dist = self._get_dist()
        obs = self._get_obs()
        
        # Reward Logic (Standardized)
        reward = -0.1 # Step penalty
        reward += (self.prev_dist - dist) * 10.0 # Progress reward
        self.prev_dist = dist
        
        terminated = False
        if dist < 0.5:
            reward += 150.0
            terminated = True
            
        return obs, reward, terminated, False, {}

    def close(self):
        p.disconnect(self.client)