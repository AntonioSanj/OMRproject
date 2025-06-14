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
dataSplit = 0.9

num_classes = 10  # Background + categories
num_epochs = 20

models_dir = modelsDir
saveModels = False

initScoreThresh = 0.05
iouThresh = 0.5
saveDataDir = frcnnPerformanceSlice
saveDataFile = '1_performance'

# random 80/20 split dataset load
full_dataset = get_coco_dataset(img_dir=images, ann_file=ann)
train_size = int(dataSplit * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print('Train and Val sizes:', len(train_dataset), len(val_dataset))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# load model and move to gpu
model = get_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# create filtered COCO object only with the images for validation
val_img_ids = [full_dataset.ids[i] for i in val_dataset.indices]
coco_gt_val = copy.deepcopy(full_dataset.coco)
coco_gt_val.dataset['images'] = [img for img in coco_gt_val.dataset['images'] if img['id'] in val_img_ids]
coco_gt_val.createIndex()

# create filtered COCO object only with the images for training
train_img_ids = [full_dataset.ids[i] for i in train_dataset.indices]
coco_gt_train = copy.deepcopy(full_dataset.coco)
coco_gt_train.dataset['images'] = [img for img in coco_gt_train.dataset['images'] if img['id'] in train_img_ids]
coco_gt_train.createIndex()

# training loop
for epoch in range(num_epochs):
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f'\nTRAINING EPOCH {epoch}')
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()

    print(f'\nEVALUATION EPOCH {epoch}')
    for inc in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        scoreThresh = initScoreThresh + inc
        print("\nScore Threshold", f"({scoreThresh:.2f})", '---------------------------------------------')

        print('Training set performance', f"(Th: {scoreThresh:.2f})\n")
        path = saveDataDir + '/train/' + saveDataFile + '_' + f"{scoreThresh:.2f}".replace('.', '_') + '.json'
        evaluate_one_epoch(model, train_loader, device, coco_gt_train, score_thresh=initScoreThresh + inc, iou_thresh=iouThresh, saveDataPath=path)

        print('\nValidation set performance', f"({scoreThresh:.2f})")
        path = saveDataDir + '/val/' + saveDataFile + '_' + f"{scoreThresh:.2f}".replace('.', '_') + '.json'
        evaluate_one_epoch(model, val_loader, device, coco_gt_val, score_thresh=initScoreThresh + inc, iou_thresh=iouThresh, saveDataPath=path)

    if saveModels:
        # Save the model's state dictionary after every epoch
        model_path = models_dir + f'fasterrcnn_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
