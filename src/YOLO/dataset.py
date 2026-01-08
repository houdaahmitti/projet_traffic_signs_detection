import os
import pandas as pd

def convert_csv_to_yolo(dataset_type="train"):
    """
    Convert CSV annotations to YOLO format for YOLOv8 training.
    dataset_type: "train" or "val"
    """

    PROJECT_DIR = "/content/drive/MyDrive/projet/projet_traffic_signs_detection"

    # Paths
    LABELS_CSV = os.path.join(PROJECT_DIR, "data/labels/sign_names.csv")
    ANN_CSV = os.path.join(PROJECT_DIR, f"data/annotations/{dataset_type}_annotations.csv")
    IMAGES_DIR = os.path.join(PROJECT_DIR, f"data/images/{dataset_type}")
    YOLO_LABELS_DIR = os.path.join(PROJECT_DIR, f"data/labels/{dataset_type}")

    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

    # Load classes
    labels_df = pd.read_csv(LABELS_CSV)
    classes = labels_df["SignName"].tolist()
    class_name_to_id = {name: idx for idx, name in enumerate(classes)}

    # Load annotations
    ann_df = pd.read_csv(ANN_CSV)

    # Convert annotations
    for file_name, group in ann_df.groupby("file_name"):
        label_file = os.path.join(
            YOLO_LABELS_DIR,
            file_name.replace(".ppm", ".txt").replace(".jpg", ".txt")
        )

        lines = []

        for _, row in group.iterrows():
            img_w = float(row["width"])
            img_h = float(row["height"])
            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])

            # YOLO format
            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h

            class_id = class_name_to_id[row["class_name"]]

            lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    print(f"✅ Conversion YOLO terminée pour {dataset_type} dataset")


# Run
convert_csv_to_yolo("train")
convert_csv_to_yolo("val")
