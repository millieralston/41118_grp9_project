import random

import cv2
import os

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"

def draw_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x, y, bw, bh = map(float, parts)

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        color = (0, 255, 0) if int(cls) == 0 else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(int(cls)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


if __name__ == "__main__":
    all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]

    print("Num images found:", len(all_images))

    if len(all_images) == 0:
        print("No images found. Check dataset path.")
        exit()

    random.shuffle(all_images)

    for img_name in all_images[:10]:
        print("Showing:", img_name)

        img_path = f"{IMG_DIR}/{img_name}"
        lbl_path = f"{LBL_DIR}/{img_name.replace('.png', '.txt')}"

        vis = draw_boxes(img_path, lbl_path)

        cv2.imshow("YOLO sanity check", vis)
        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()