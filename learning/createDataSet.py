import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from vision.ImageOperations import loadImageGrey


class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)

        # Load the image
        image, _ = loadImageGrey(image_path)

        image = torch.from_numpy(image).float()

        if image.ndimension() == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)  # Rearrange dimensions to (C, H, W)

        # Load annotations
        csv_file = os.path.join(self.annotation_dir, image_file.replace(".png", ".csv"))
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Annotation file {csv_file} not found for image {image_file}")

        annotations = pd.read_csv(csv_file)
        boxes, labels = [], []

        for _, row in annotations.iterrows():
            x1 = row["x1"]
            x2 = row["x2"]
            y1 = row["y1"]
            y2 = row["y2"]
            boxes.append([x1, y1, round(x2-x1), round(y2-y1)])
            labels.append(
                {
                    "One": 1, "Double": 2, "Four": 3, "Half": 4, "Quarter": 5
                    # "GClef": 6, "FClef": 7, "OpeningBracket": 8, "RestOne": 9, "RestHalf": 10,
                }
                [row["classTitle"]]
            )
        print(boxes)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        return image, target
