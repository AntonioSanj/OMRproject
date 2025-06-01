import copy

import torch
from torch.utils.data import DataLoader, random_split

from constants import *
from learning.FasterRCNN.getModel import get_model
from learning.utils.cocoDataSet import get_coco_dataset
from learning.utils.evalEpoch import evaluate_one_epoch
from learning.utils.trainEpoch import train_one_epoch

images = mySlicedDataImg
ann = mySlicedDataCoco
models_dir = modelsDir
num_classes = 10  # Background + categories
num_epochs = 20
saveModels = False

full_dataset = get_coco_dataset(img_dir=images, ann_file=ann)

# random 80/20 split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print('Train and Val sizes:', len(train_dataset), len(val_dataset))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


model = get_model(num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Create filtered COCO object only with validation image IDs
val_img_ids = [full_dataset.ids[i] for i in val_dataset.indices]
coco_gt = copy.deepcopy(full_dataset.coco)
coco_gt.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in val_img_ids]
coco_gt.createIndex()

# Training loop
for epoch in range(num_epochs):
    print(f'\nTRAINING EPOCH {epoch} -----------------------------------------------------')
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()
    print(f'\nVALIDATION EPOCH {epoch} -----------------------------------------------------')
    evaluate_one_epoch(model, val_loader, device, coco_gt)
    if saveModels:
        # Save the model's state dictionary after every epoch
        model_path = models_dir + f'fasterrcnn_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
