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

        # Ensure image is converted to a PyTorch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()  # Convert numpy array to float tensor

        # If the image is grayscale, repeat it across 3 channels
        if image.ndimension() == 2:  # Single-channel image
            image = image.unsqueeze(0).repeat(3, 1, 1)  # Convert to 3-channel image

        # Ensure image is in (C, H, W) format
        if image.ndimension() == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)  # Rearrange dimensions to (C, H, W)

        # Load annotations
        csv_file = os.path.join(self.annotation_dir, image_file.replace(".png", ".csv"))
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Annotation file {csv_file} not found for image {image_file}")

        annotations = pd.read_csv(csv_file)
        boxes, labels = [], []

        for _, row in annotations.iterrows():
            boxes.append([row["x1"], row["y1"], row["x2"], row["y2"]])
            labels.append(
                {"GClef": 1, "FClef": 2, "Half": 3, "One": 4, "Double": 5,
                 "RestOne": 6, "RestHalf": 7, "Four": 8, "OpeningBracket": 9, "Quarter": 10}
                [row["classTitle"]]
            )

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        return image, target
