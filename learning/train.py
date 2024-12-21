from random import seed
from constants import *

import torch
from torch.utils.data import DataLoader, random_split

from learning.createDataSet import MyDataset
from learning.modelLoader import get_model
from learning.testImage import testImage
from learning.trainingMetrics import calculate_accuracy

seed(1)

# Hyperparameters
num_classes = 6  # 10 classes + background
num_epochs = 10
learning_rate = 0.005

# paths
image_dir = myDataImg
annotation_dir = myDataCsv2

# dataset load
dataset = MyDataset(image_dir, annotation_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
# model load
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model(num_classes)
model.to(device)

# optimizer and learning rate scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# training
model.train()  # Make sure the model is in training mode
for epoch in range(num_epochs):
    total_precision = 0
    total_recall = 0
    num_batches = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # evaluation mode
        model.eval()
        with torch.no_grad():
            # get predictions from the model
            predictions = model(images)

        model.train()

        # compute precision and recall
        precision, recall = calculate_accuracy(predictions, targets)

        # add to epoch total
        total_precision += precision
        total_recall += recall
        num_batches += 1

    # compute average
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    print(f"Epoch {epoch + 1}, Loss: {losses.item():.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # Step the learning rate scheduler
    lr_scheduler.step()

print("\nFINISHED\n")

testImage(feelTheLove, model, device)
# torch.save(model.state_dict(), "fasterrcnn_model.pth")
