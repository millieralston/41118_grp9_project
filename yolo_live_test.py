import cv2
import numpy as np
from ultralytics import YOLO
from husky_chaser_env import HuskyChaserEnv

"""
Real-time YOLO inference demo on the simulation environment.

This script loads a trained YOLO model and runs live inference on
camera frames from the PyBullet chaser environment. The detected
runner is visualised in real time using bounding boxes overlaid on
the rendered image.

The script is used for qualitative demonstration of model performance
in a live simulation setting for demonstration purposes only, and is not
used in the reinforcement learning training loop.
"""

model = YOLO("runs/detect/yolo_training/runner_detector_v2/weights/best.pt")

env = HuskyChaserEnv(render_mode="gui", runner_spawn_mode="front")
obs, _ = env.reset()

while True:

    img = env.get_camera_image_yolo()

    print(img.shape, img.dtype, img.min(), img.max())

    # --- FIX IMAGE FORMAT ---
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    results = model(img)[0]
    annotated = results.plot()

    annotated = cv2.resize(annotated, (960, 720))

    cv2.imshow("YOLO Live", annotated)

    key = cv2.waitKey(1)
    if key == 27:
        break
