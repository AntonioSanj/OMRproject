import os

import numpy as np
import torch

from vision.ImageOperations import loadImageGrey


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir):
        # directory of images
        self.images_dir = images_dir
        # directory of ground truth
        self.masks_dir = masks_dir

        # array with all the image file names
        self.imgs = list(sorted(os.listdir(self.images_dir)))

    def __getitem__(self, i):
        # Load image
        img_path = os.path.join(self.images_dir, self.imgs[i])
        img, _ = loadImageGrey(img_path)

        # Convert the image from NumPy array to PyTorch tensor
        img = torch.as_tensor(img, dtype=torch.float32)  # Convert to float tensor

        # Check if the image is grayscale (1 channel)
        if img.ndimension() == 2:  # If shape is (H, W)
            img = img.unsqueeze(0)  # Add channel dimension
            img = img.repeat(3, 1, 1)  # Convert to 3 channels
        elif img.shape[0] == 1:  # If only 1 channel (grayscale)
            img = img.repeat(3, 1, 1)  # Convert to 3 channels

        # Ensure the image shape is (C, H, W)
        img = img.permute(2, 0, 1)  # Change from (H, W, C) to (C, H, W)

        # Construct mask path
        mask_name = os.path.splitext(self.imgs[i])[0] + "_inst.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Check if mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        _, grayMask = loadImageGrey(mask_path)

        # Convert mask to numpy array for processing
        grayMask = np.array(grayMask)

        # Get unique object IDs (ignoring background with ID 0)
        obj_ids = np.unique(grayMask)
        obj_ids = obj_ids[obj_ids != 0]

        # Create binary masks and bounding boxes for each object ID
        masks = grayMask == obj_ids[:, None, None]

        # Compute bounding boxes from masks
        # Compute bounding boxes from masks
        boxes = []
        for j in range(len(obj_ids)):
            pos = np.where(masks[j])
            if pos[0].size == 0:  # Skip masks with no valid pixels
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # Check if bounding box is valid
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Ensure valid masks match valid boxes
        valid_indices = torch.nonzero(boxes.sum(dim=1) > 0).squeeze()
        if valid_indices.numel() > 0:  # Only filter if there are valid indices
            boxes = boxes[valid_indices]
            masks = masks[valid_indices]
        else:
            raise ValueError("No valid masks or boxes found in the sample.")

        # Assuming all objects are of class 1 (adjust as needed)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        return img, target

    def __len__(self):
        return len(self.imgs)
