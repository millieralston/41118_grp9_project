import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

"""
Vision-based perception module for integrated CNN + PPO training experiments.

This module extends the standard perception pipeline by replacing the
ground-truth runner position with detections obtained from a YOLO object
detector. Runner coordinates are extracted from camera images, normalised
to the camera frame, and combined with visibility and confidence metrics
to form a 10-dimensional observation vector.

The module was developed to investigate vision-based reinforcement learning
using camera observations rather than simulator ground-truth data.
"""

# describes the observation space format (a vector with 10 floating point values)
def create_observation_space():
    # observation space vector layout
    # [runner_x, runner_y, runner_visible, runner_confidence, obstacle1_x, obstacle1_y, obstacle2_x, obstacle2_y, obstacle3_x, obstacle3_y]

    return spaces.Box(
        low=-20,
        high=20,
        # shape=(8,),
        shape=(10,),
        dtype=np.float32
    )

# returns the observation space vector for the current state of the environment (positions of runner and obstacles relative to the chaser)
def get_observation(env):
    observation = []
    debug = getattr(env, "debug_perception", False)

    env.use_yolo = True # toggle to switch between retrieving runner position from YOLO vs simulation coordinates

    chaser_pos, chaser_orn = p.getBasePositionAndOrientation(env.chaser)

    runner_pos, _ = p.getBasePositionAndOrientation(env.runner)

    if debug:
         env.runner_text_id =p.addUserDebugText(
            "RUNNER",
            runner_pos,
            textColorRGB=[1, 0, 0],
            replaceItemUniqueId=env.runner_text_id
        )

    if debug:
        env.runner_line_id =p.addUserDebugLine(
            chaser_pos,
            runner_pos,
            [1, 0, 0],   # red
            lineWidth=3,
            replaceItemUniqueId=env.runner_line_id
        )

    if env.use_yolo:
        runner_x, runner_y = get_yolo_runner(env)

    else:
        runner_x, runner_y = get_relative_position(
            chaser_pos,
            chaser_orn,
            runner_pos
        )

    observation.extend([runner_x, runner_y, float(env.runner_visible), float(env.runner_confidence)])

    closest_obstacles = get_closest_obstacles(env)

    while len(env.obstacle_line_ids) < len(closest_obstacles):
        env.obstacle_line_ids.append(-1)

    while len(env.obstacle_text_ids) < len(closest_obstacles):
        env.obstacle_text_ids.append(-1)

    for i, (_, obs_x, obs_y, obs_world_pos) in enumerate(closest_obstacles):
        
        if debug:
            env.obstacle_line_ids[i] = p.addUserDebugLine(
                chaser_pos,
                obs_world_pos,
                [0, 1, 0],   # green
                lineWidth=2,
                replaceItemUniqueId=env.obstacle_line_ids[i]
            )

            env.obstacle_text_ids[i] = p.addUserDebugText(
                f"OBS {i+1}",
                obs_world_pos,
                textColorRGB=[0, 1, 0],
                replaceItemUniqueId=env.obstacle_text_ids[i]
            )

        observation.extend([obs_x, obs_y])

    while len(observation) < 10:
        observation.append(0.0)
    
    return np.array(observation, dtype=np.float32)

# function to calculate the relative position of an object to a target (the chaser)
def get_relative_position(reference_pos, reference_orn, target_pos):
    inv_pos, inv_orn = p.invertTransform(reference_pos, reference_orn)

    target_orn = [0, 0, 0, 1]

    rel_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, target_pos, target_orn)

    return rel_pos[0], rel_pos[1]

# computing relative positions of all obstacles to the chaser, returns the 3 closest ones
def get_closest_obstacles(env, max_obstacles=3):
    chaser_pos, chaser_orn = p.getBasePositionAndOrientation(env.chaser)

    obstacle_data = []

    for obstacle_pos in env.obstacle_positions:

        obstacle_world_pos = [
            obstacle_pos[0],
            obstacle_pos[1],
            0.0
        ]

        rel_x, rel_y = get_relative_position(
            chaser_pos,
            chaser_orn,
            obstacle_world_pos
        )

        distance = np.sqrt(rel_x**2 + rel_y**2)

        obstacle_data.append((distance, rel_x, rel_y, obstacle_world_pos))

    obstacle_data.sort(key=lambda x: x[0])

    return obstacle_data[:max_obstacles]

def get_yolo_runner(env):
    env.yolo_step_counter += 1

    # Only run YOLO every N steps
    if env.yolo_step_counter % env.yolo_skip == 0:

        img = env.get_camera_image_yolo()
        results = env.yolo_model.predict(img, verbose=False)[0]

        best_box = None
        best_area = 0

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])

        # if runner visible
        if best_box is not None:
            x1, y1, x2, y2 = best_box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            h, w = img.shape[:2]

            runner_x = (cx / w) * 2 - 1
            runner_y = (cy / h) * 2 - 1

            new_xy = np.array([runner_x, runner_y], dtype=np.float32)

            # initialize cache if needed
            if not hasattr(env, "yolo_cache"):
                env.yolo_cache = new_xy.copy()
            else:
                env.yolo_cache[:] = new_xy

            env.runner_visible = True
            env.runner_confidence = min(env.runner_confidence + 0.2, 1.0)

        # if runner not visible
        else:
            env.runner_visible = False
            env.runner_confidence *= 0.9

    # ALWAYS return last known value
    return env.yolo_cache[0], env.yolo_cache[1]