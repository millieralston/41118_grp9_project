from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/yolo_training/runner_detector_v1/weights/best.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,
        workers=2,
        project="yolo_training",
        name="runner_detector_v2"
    )

if __name__ == "__main__":
    main()