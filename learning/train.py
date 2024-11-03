from random import seed

import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from learning.createDataSet import CustomDataset
from learning.modelLoader import model

seed(1)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RUNNING IN ", device, "\tCuda version: ", torch.version.cuda)

# Move model to device
model.to(device)

# hyperparameters
batch_size = 2
num_epochs = 10
learning_rate = 0.005

# paths to your images and annotation file
images_dir = r"C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\denseDataSet\images"
annotations = r"C:\Users\Usuario\Desktop\UDC\QUINTO\TFG\src_code\dataset\denseDataSet\annotations"

# initialize dataset
dataset = CustomDataset(images_dir, annotations)

# set size for train and validation set
trainSet_size = int(0.8 * len(dataset))
validationSet_size = len(dataset) - trainSet_size

# assign items randomly to a set
trainSet, validationSet = random_split(dataset, [trainSet_size, validationSet_size])

train_loader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(validationSet, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

print("STARTING TRAINING...")
# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        print("inn loop")
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
print("STARTING EVALUATION...")
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        output = model(images)
        # Calculate and track metrics (e.g., mAP, recall)

# torch.save(model.state_dict(), "faster_rcnn_model.pth")
