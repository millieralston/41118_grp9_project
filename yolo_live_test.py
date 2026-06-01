import cv2
import numpy as np
from ultralytics import YOLO
from husky_chaser_env import HuskyChaserEnv

model = YOLO("runs/detect/yolo_training/runner_detector_v1/weights/best.pt")

env = HuskyChaserEnv(render_mode="gui")
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