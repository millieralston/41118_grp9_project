from ultralytics import YOLO

"""
YOLO training script for runner detection.

This script fine-tunes a pre-trained YOLOv11n model on the custom
synthetic dataset generated from the PyBullet simulation. It configures
training parameters such as image size, batch size, number of epochs,
and early stopping to optimise detection performance.

The resulting model is saved under the specified project directory for
later evaluation and deployment.
"""

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        workers=2,
        project="yolo_training",
        name="runner_detector_v2"
    )

if __name__ == "__main__":
    main()