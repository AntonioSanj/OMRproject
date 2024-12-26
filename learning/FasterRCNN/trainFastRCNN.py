import torch
from torch.utils.data import DataLoader

from constants import *
from learning.FasterRCNN.cocoDataSet import get_coco_dataset
from learning.FasterRCNN.getModel import get_model
from learning.FasterRCNN.trainEpoch import train_one_epoch

train_dataset = get_coco_dataset(
    img_dir=myDataImg,
    ann_file=myDataCoco
)

val_dataset = get_coco_dataset(
    img_dir=myDataImg,
    ann_file=myDataCoco
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

num_classes = 11  # Background + categories
model = get_model(num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()

    # Save the model's state dictionary after every epoch
    model_path = f'fasterrcnn_resnet50_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
