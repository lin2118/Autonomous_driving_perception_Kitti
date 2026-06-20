import os
import cv2

BASE_DIR = os.path.expanduser("~/kitti_yolo/dataset")

LABEL_DIR = os.path.join(BASE_DIR, "kitti_labels/data_object_label_2/training/label_2")
IMAGE_DIR = os.path.join(BASE_DIR, "raw_kitti/training/image_2")
OUTPUT_DIR = os.path.join(BASE_DIR, "labels/labels_all")

os.makedirs(OUTPUT_DIR, exist_ok=True)

classes = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Van": 3,
    "Truck": 4,
    "Tram": 5,
    "Person_sitting": 6,
    "Misc": 7,
    "DontCare": 8
}

def convert_bbox(w, h, x1, y1, x2, y2):
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh

count = 0

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(LABEL_DIR, file)
    img_path = os.path.join(IMAGE_DIR, file.replace(".txt", ".png"))

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape
    yolo_lines = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 8:
                continue

            class_name = parts[0]

            if class_name not in classes:
                continue

            class_id = classes[class_name]

            x1, y1, x2, y2 = map(float, parts[4:8])

            xc, yc, bw, bh = convert_bbox(w, h, x1, y1, x2, y2)

            yolo_lines.append(f"{class_id} {xc} {yc} {bw} {bh}")

    out_path = os.path.join(OUTPUT_DIR, file)

    with open(out_path, "w") as f:
        f.write("\n".join(yolo_lines))

    count += 1

print(f"Converted {count} files")