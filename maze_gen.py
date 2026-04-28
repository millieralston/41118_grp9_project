import pybullet as p
import pybullet_data
import random
import time

# --- Configuration ---
MAP_WIDTH = 11   # Must be odd
MAP_HEIGHT = 11  # Must be odd
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.9 # Slightly less than 1.0 to see grid lines if desired
CELL_SIZE = 1.0

def generate_maze(w, h):
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

def setup_pybullet_maze(maze):
    # Initialize PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load Ground Plane
    p.loadURDF("plane.urdf")

    # Define the visual and collision shape for a wall block
    # Half-extents are used, so 0.5 creates a 1.0 meter block
    wall_col_id = p.createCollisionShape(p.GEOM_BOX, 
                                        halfExtents=[CELL_SIZE/2, CELL_SIZE/2, WALL_HEIGHT/2])
    wall_vis_id = p.createVisualShape(p.GEOM_BOX, 
                                       halfExtents=[CELL_SIZE/2, CELL_SIZE/2, WALL_HEIGHT/2],
                                       rgbaColor=[0.7, 0.7, 0.7, 1])

    # Instantiate walls
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == 1:
                x_pos = c * CELL_SIZE
                y_pos = r * CELL_SIZE
                z_pos = WALL_HEIGHT / 2
                
                p.createMultiBody(baseMass=0, # Mass 0 makes it static
                                  baseCollisionShapeIndex=wall_col_id,
                                  baseVisualShapeIndex=wall_vis_id,
                                  basePosition=[x_pos, y_pos, z_pos])

    return physicsClient

if __name__ == "__main__":
    maze_layout = generate_maze(MAP_WIDTH, MAP_HEIGHT)
    client = setup_pybullet_maze(maze_layout)
    
    print("Maze generated in PyBullet. Close the GUI or Ctrl+C to exit.")
    
    # Keep the simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()