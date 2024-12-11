from random import seed
from constants import *

import torch
from torch.utils.data import DataLoader, random_split

from learning.createDataSet import MyDataset
from learning.modelLoader import get_model

seed(1)

# Hyperparameters
num_classes = 11  # 3 classes + background
num_epochs = 10
learning_rate = 0.005

# Paths
image_dir = myDataImg
annotation_dir = myDataCsv

# Dataset and DataLoader
dataset = MyDataset(image_dir, annotation_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# Model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model(num_classes)
model.to(device)

# Optimizer and Learning Rate Scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training Loop
model.train()
for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {losses.item():.4f}")

print("\nFINISHED\n")
# torch.save(model.state_dict(), "fasterrcnn_model.pth")
