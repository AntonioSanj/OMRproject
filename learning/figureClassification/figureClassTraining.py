import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from constants import *

# Directory
data_dir = myFiguresDataSet  # Your dataset folder path

# Hyperparameters
batch_size = 4
learning_rate = 0.005
validation_split = 0.2  # Percentage of data to use for validation
patience = 500
max_epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the full dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(full_dataset.classes)  # Number of class folders

# Split dataset into training and validation
dataset_size = len(full_dataset)
val_size = int(validation_split * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Dataloader dictionary for easier looping
dataloaders = {
    "train": train_loader,
    "val": val_loader
}

# Load ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
epochs_without_improvement = 0

stop = False
epoch = 0

# Training loop
while not stop:
    print(f"Epoch {epoch + 1}:")

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        if phase == "train":
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_acc.item())

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        print(f"{phase}\t Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%")

    epoch += 1

    if epochs_without_improvement >= patience:
        print(f"TRAINING STOPPED. {patience} epochs with no improvement.")
        stop = True
    if epoch >= max_epochs:
        print(f"\nTRAINING STOPPED. Max epoch number reached: {max_epochs}.")
        stop = True
    print("")

torch.save(model.state_dict(), figureModels + f"figure_classification_model{epoch}.pth")

performance_data = {
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies,
    "train_losses": train_losses,
    "val_losses": val_losses
}

with open(figuresPerformanceDataJson, "w") as f:
    json.dump(performance_data, f, indent=4)

print(f"Performance data saved")
