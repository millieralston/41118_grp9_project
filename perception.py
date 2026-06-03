import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

# describes the observation space format (a vector with 8 floating point values)
def create_observation_space():
    # observation space vector layout
    # [runner_x, runner_y, obstacle1_x, obstacle1_y, obstacle2_x, obstacle2_y, obstacle3_x, obstacle3_y]

    return spaces.Box(
        low=-20,
        high=20,
        shape=(8,),
        dtype=np.float32
    )

# returns the observation space vector for the current state of the environment (positions of runner and obstacles relative to the chaser)
def get_observation(env):
    observation = []
    debug = getattr(env, "debug_perception", False)

    chaser_pos, chaser_orn = p.getBasePositionAndOrientation(env.chaser)

    runner_pos, _ = p.getBasePositionAndOrientation(env.runner)

    if debug:
        p.addUserDebugText(
            "RUNNER",
            runner_pos,
            textColorRGB=[1, 0, 0],
            lifeTime=0.1
        )

        p.addUserDebugLine(
            chaser_pos,
            runner_pos,
            [1, 0, 0],   # red
            lineWidth=3,
            lifeTime=0.1
        )

    runner_x, runner_y = get_relative_position(chaser_pos, chaser_orn, runner_pos)

    # if debug:
    #     print(f"Runner pos: ({runner_x:.2f}, {runner_y:.2f}) | ")

    observation.extend([runner_x, runner_y])

    closest_obstacles = get_closest_obstacles(env)

    while len(env.obstacle_line_ids) < len(closest_obstacles):
        env.obstacle_line_ids.append(-1)

    while len(env.obstacle_text_ids) < len(closest_obstacles):
        env.obstacle_text_ids.append(-1)

    # added debug visualization of the closest obstacles relative to the chaser, 
    # which can help understand what the agent is perceiving and how it relates to its actions.
    for i, (_, obs_x, obs_y, obs_world_pos) in enumerate(closest_obstacles):
        if debug:
            # print(f"Obs pos: ({obs_x:.2f}, {obs_y:.2f})")
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

        observation.extend([obs_x, obs_y]) # relative position of the obstacle to the chaser, which is crucial for the agent to learn how to navigate around obstacles while pursuing the runner.

    while len(observation) < 8:
        observation.extend([0.0, 0.0])

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
    """Returns the runner position estimate used by the YOLO demo overlay."""
    if getattr(env, "yolo_model", None) is None:
        return env._relative_runner_position()

    if getattr(env, "yolo_step_counter", 0) % getattr(env, "yolo_skip", 1) == 0:
        try:
            image = env.get_camera_image_yolo()
            result = env.yolo_model(image, verbose=False)[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                center_x = float((xyxy[0] + xyxy[2]) * 0.5)
                center_y = float((xyxy[1] + xyxy[3]) * 0.5)
                width = max(float(image.shape[1]), 1.0)
                height = max(float(image.shape[0]), 1.0)
                env.yolo_cache = np.array(
                    [
                        (center_x / width - 0.5) * 2.0,
                        (center_y / height - 0.5) * 2.0,
                    ],
                    dtype=np.float32,
                )
                env.runner_visible = True
                env.runner_confidence = float(boxes.conf[best_idx].item())
            else:
                env.runner_visible = False
                env.runner_confidence = 0.0
        except Exception:
            env.runner_visible = False
            env.runner_confidence = 0.0

    env.yolo_step_counter = getattr(env, "yolo_step_counter", 0) + 1
    return float(env.yolo_cache[0]), float(env.yolo_cache[1])
