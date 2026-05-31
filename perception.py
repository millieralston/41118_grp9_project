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
        env.runner_text_id =p.addUserDebugText(
            "RUNNER",
            runner_pos,
            textColorRGB=[1, 0, 0],
            replaceItemUniqueId=env.runner_text_id
        )

        env.runner_line_id =p.addUserDebugLine(
            chaser_pos,
            runner_pos,
            [1, 0, 0],   # red
            lineWidth=3,
            replaceItemUniqueId=env.runner_line_id
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

    if debug:
        for i, (_, obs_x, obs_y, obs_world_pos) in enumerate(closest_obstacles):
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

            observation.extend([obs_x, obs_y])

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
