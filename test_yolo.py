from tkinter.font import names

from ultralytics import YOLO
import cv2
import glob

"""
YOLO evaluation and visualisation script.

This script evaluates a trained YOLO model on the validation dataset
and visualises prediction performance. It first computes validation
metrics (e.g., precision, recall, and mAP), then runs inference on
validation images and overlays predicted bounding boxes and confidence
scores.

The tool is used for quantitative and qualitative assessment of detection performance
and to verify that the trained model generalises correctly to unseen
simulation frames.
"""

IMG_DIR = "dataset/images/val"

model = YOLO("runs/detect/yolo_training/runner_detector_v2/weights/best.pt")

def draw_predictions(img, results):

    for r in results:

        boxes = r.boxes

        for b in boxes:

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            names = model.names
            label = names[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    return img


if __name__ == "__main__":
    metrics = model.val(data="dataset.yaml")
    print(metrics.results_dict)

    image_paths = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)

        results = model(img)
        vis = draw_predictions(img, results)

        cv2.imshow("YOLO predictions", vis)

        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
