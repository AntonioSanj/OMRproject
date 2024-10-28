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

        # Load mask in gray scale
        mask_path = os.path.join(self.masks_dir, self.imgs[i], "_inst")  # mask_path = imagePath+"_inst"
        _, grayMask = loadImageGrey(mask_path)

        # Convert mask to numpy array for processing
        grayMask = np.array(grayMask)

        # Get unique object IDs (ignoring background with ID 0)
        obj_ids = np.unique(grayMask)
        obj_ids = obj_ids[obj_ids != 0]

        # Split mask into binary masks for each instance
        masks = grayMask == obj_ids[:, None, None]

        # Compute bounding boxes from masks
        boxes = []
        for j in range(len(obj_ids)):
            pos = np.where(masks[j])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),),
                            dtype=torch.int64)  # Assuming all objects are of class 1 (change as needed)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        return img, target

    def __len__(self):
        return len(self.imgs)
