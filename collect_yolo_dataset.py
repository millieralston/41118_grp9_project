from husky_chaser_env import HuskyChaserEnv
import cv2
import os
import pybullet as p
import numpy as np

from perception import get_relative_position

env = HuskyChaserEnv(render_mode="gui", runner_spawn_mode="front")

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"
SEG_DIR = "dataset/seg_images"
REJECT_DIR = "dataset/rejected"

EDGE_IMG_DIR = "dataset/edge_case/images"
EDGE_LBL_DIR = "dataset/edge_case/labels"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

os.makedirs(SEG_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)

os.makedirs(EDGE_IMG_DIR, exist_ok=True)
os.makedirs(EDGE_LBL_DIR, exist_ok=True)

# collects relevant data from the camera image used to train yolo
def get_camera(env, width=640, height=480):
    pos, orn = p.getBasePositionAndOrientation(env.chaser)

    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]

    cam_pos = [pos[0] + forward[0]*0.5,
               pos[1] + forward[1]*0.5,
               pos[2] + 0.6]

    target = [pos[0] + forward[0]*3,
              pos[1] + forward[1]*3,
              pos[2] + 0.4]

    view_matrix = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)

    width, height, rgb, depth, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        flags = (p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX),
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # print("Unique seg IDs:", np.unique(seg))
    # print("Runner ID in seg?:", env.runner in np.unique(seg))

    rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
    seg = np.reshape(seg, (height, width))

    return rgb.astype(np.uint8), seg

# converting segmentation mask to yolo bounding boxes
def seg_to_bboxes(seg, object_id_map):
    labels = []

    for obj_id, class_id in object_id_map.items():

        mask = seg == obj_id
        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        x_center = (x_min + x_max) / 2 / seg.shape[1]
        y_center = (y_min + y_max) / 2 / seg.shape[0]
        width = (x_max - x_min) / seg.shape[1]
        height = (y_max - y_min) / seg.shape[0]

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return labels

def get_runner_seg_ids(env, seg):
    """
    Finds all segmentation IDs that belong to the runner.
    This avoids assuming env.runner == seg ID.
    """

    runner_mask = np.zeros_like(seg, dtype=bool)

    object_ids = seg & ((1 << 24) - 1)

    runner_mask = (object_ids == env.runner)

    # If that fails, we infer from presence in scene via bbox fallback
    if not np.any(runner_mask):
       print("Runner not visible in segmentation image.")

    # print(f"Runner pixels found: {np.count_nonzero(runner_mask)}")

    return runner_mask

if __name__ == "__main__":
    target_samples = 3500
    saved = 0
    accepted = 0
    rejected = 0

    while saved < target_samples: # collect 2,500 samples
        print(f"\n===== IMAGE {saved} =====")

        env.reset()

        runner_pos, _ = p.getBasePositionAndOrientation(env.runner)
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(env.chaser)

        dist = np.linalg.norm(np.array(runner_pos[:2]) - np.array(chaser_pos[:2]))

        # is_edge_case = dist < 4.0 or dist > 14.0

        # if dist < 3.0:
        #     continue  # reject sample, try again

        # get camera data
        rgb, seg = get_camera(env)

        # DEBUGGING PRINT STATEMENTS
        # rx, ry = get_relative_position(
        #     chaser_pos,
        #     chaser_orn,
        #     runner_pos
        # )

        # print("Runner pos:", runner_pos)
        # print("Chaser pos:", chaser_pos)
        # print(f"Runner relative position: ({rx:.2f}, {ry:.2f})")

        # print("Unique seg IDs:", np.unique(seg))

        # for sid in np.unique(seg):
        #     if sid < 0:
        #         continue

        #     object_id = sid & ((1 << 24) - 1)
        #     link_index = (sid >> 24) - 1

        #     print(
        #         f"seg={sid}, object={object_id}, link={link_index}"
        #     )

        runner_mask = get_runner_seg_ids(env, seg)

        print("Runner pixels:", np.count_nonzero(runner_mask))

        # print(f"[{i}] Runner body ID:", env.runner)

        object_ids = seg & ((1 << 24) - 1)

        print("Runner hit:", np.any(object_ids == env.runner))

        # save rgb image
        # if is_edge_case:
        #     img_path = f"{EDGE_IMG_DIR}/{saved}.png"
        # else:
        #     img_path = f"{IMG_DIR}/{saved}.png"

        img_path = f"{IMG_DIR}/{saved}.png"
        cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # # save segmentation image for debugging
        # seg_vis = (seg.astype(np.float32) - seg.min())
        # seg_vis /= seg_vis.max() + 1e-6
        # seg_vis = (seg_vis * 255).astype(np.uint8)

        # seg_path = f"{SEG_DIR}/{saved}.png"
        # cv2.imwrite(seg_path, seg_vis)

        # # obstacles still use segmentation IDs - removing for now as we're focusing on runner detection, but we can add back later if needed
        # object_id_map = {}

        # for obs_id in env.obstacle_body_ids:
        #     object_id_map[obs_id] = 1

        # # save labels in yolo format (class_id, x_center, y_center, width, height)
        # labels = seg_to_bboxes(seg, object_id_map)

        ys, xs = np.where(runner_mask)

        if len(xs) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            bbox_area = (x_max - x_min) * (y_max - y_min)
            visible_area = len(xs)

            # DEBUG PRINTS
            print("bbox_area:", bbox_area)
            print("visible_area:", visible_area)

            # HARD FILTER 1: tiny fragments FIRST
            if visible_area < 200:
                cv2.imwrite(
                    f"{REJECT_DIR}/{saved}.png",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                labels = []
            else:

                # only compute ratio if object is big enough
                if bbox_area < 1:
                    labels = []
                else:
                    visibility_ratio = visible_area / (bbox_area + 1e-6)
                    print("ratio:", visibility_ratio)

                    # HARD FILTER 2: bbox sanity
                    if bbox_area < 80:
                        labels = []

                    elif visibility_ratio < 0.5:
                        cv2.imwrite(
                            f"{REJECT_DIR}/{saved}.png",
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        )
                        labels = []

                    else:
                        x_center = (x_min + x_max) / 2 / seg.shape[1]
                        y_center = (y_min + y_max) / 2 / seg.shape[0]
                        width = (x_max - x_min) / seg.shape[1]
                        height = (y_max - y_min) / seg.shape[0]

                        labels = [f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"]
        
                accepted += 1 if labels else 0
                rejected += 1 if not labels else 0

        else:
            labels = [] # no runner visible, no labels

        with open(f"{LBL_DIR}/{saved}.txt", "w") as f:
            for l in labels:
                f.write(l + "\n")

        saved += 1

        print("accepted:", accepted)
        print("rejected:", rejected)
        print(f"Saved {saved}/{target_samples}")
