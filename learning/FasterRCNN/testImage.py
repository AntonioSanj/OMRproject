import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import functional as F

from constants import modelsDir, myDataImg, mySlicedDataImg, slicedModelsDir, fastRCNNOutput, fullsheetsDir
from learning.FasterRCNN.getModel import get_model

COCO_CLASSES = {0: "Background", 1: "One", 2: "Double", 3: "Four", 4: "Half", 5: "Quarter", 6: "GClef", 7: "FClef",
                8: "RestOne", 9: "RestHalf"}


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")


def draw_boxes(image, prediction, threshold=0.0, fig_size=(10, 10), saveDir=None):
    # prediction contains:
    # - boxes: predicted bounding boxes
    # - labels: predicted class labels
    # - scores: predicted scores for each box (confidence level)
    boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
    scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores

    print(labels)

    width, height = image.size
    dpi = 100
    fig_size = (width / dpi, height / dpi)
    fig, ax = plt.subplots(1, figsize=fig_size, dpi=dpi)
    ax.imshow(image)

    print('drawing data')
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=1, edgecolor='r', facecolor='none'))
            ax.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')

    ax.axis('off')
    if saveDir is not None:
        plt.savefig(saveDir, bbox_inches='tight', pad_inches=0)
        print(f'Image saved at {saveDir}')
    plt.show()


def testImage(image_path, modelDir, num_classes, threshold=0.0, saveDir=None):
    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(modelDir))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    image = Image.open(image_path).convert("RGB")  # Open image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(image_tensor)

    draw_boxes(Image.open(image_path), prediction, threshold, fig_size=(9, 16), saveDir=saveDir)  # Example of increased size


# testImage(fullsheetsDir + '/thinking_out_loud1.png', modelsDir + '/fasterrcnn_epoch_10.pth', 10, 0.01, fastRCNNOutput + '/img2.png')
# testImage(myDataImg + '/feelthelove1.png', modelsDir + '/fasterrcnn_epoch_10.pth', 10, 0.01, fastRCNNOutput + '/img1.png')
# testImage(mySlicedDataImg + '/slice30.png', slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10, 0.15)
# testImage(mySlicedDataImg + '/slice31.png', slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10, 0.15,fastRCNNOutput+'img3.png')
# testImage(mySlicedDataImg + '/slice32.png', slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10, 0.15, fastRCNNOutput+'img4.png')
# testImage(mySlicedDataImg + '/slice33.png', slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10, 0.15, fastRCNNOutput+'img5.png')
# testImage(mySlicedDataImg + '/slice36.png', slicedModelsDir + 'fasterrcnn_epoch_6.pth', 10, 0.15, fastRCNNOutput+'img6.png')
