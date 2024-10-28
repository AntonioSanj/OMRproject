from random import seed

import torch
from torch import device
from torch.utils.data import DataLoader, random_split

from learning.createDataSet import CustomDataset
from learning.modelLoader import model

seed(1)

# Hyperparameters
batch_size = 2
num_epochs = 10
learning_rate = 0.005

# Paths to your images and annotation file
images_dir = r"C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\denseDataSet\images"
annotations = r"C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\denseDataSet\annotations"

# Create dataset
dataset = CustomDataset(images_dir, annotations)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calculate the losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item()}")

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        output = model(images)
        # Calculate and track metrics (e.g., mAP, recall)

# torch.save(model.state_dict(), "faster_rcnn_model.pth")
