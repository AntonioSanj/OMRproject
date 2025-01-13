import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import functional as F

from constants import *
from learning.FasterRCNN.getModel import get_model


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")


COCO_CLASSES = {0: "Background", 1: "One", 2: "Double", 3: "Four", 4: "Half", 5: "Quarter", 6: "GClef", 7: "FClef",
                8: "RestOne", 9: "RestHalf"}

num_classes = 6

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the trained model
model = get_model(num_classes)
model.load_state_dict(torch.load(modelsDir + 'fasterrcnn_epoch_10.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the unseen image
image_path = feelTheLove
image = Image.open(image_path).convert("RGB")  # Open image
image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
image_tensor = image_tensor.to(device)

with torch.no_grad():  # Disable gradient computation for inference
    prediction = model(image_tensor)


# `prediction` contains:
# - boxes: predicted bounding boxes
# - labels: predicted class labels
# - scores: predicted scores for each box (confidence level)


# Draw bounding boxes with the correct class names and increase image size
def draw_boxes(image, prediction, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
    print(boxes)
    labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
    scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores

    # Set a threshold for showing boxes (e.g., score > 0.5)
    threshold = 0.0

    # Set up the figure size to control the image size
    plt.figure(figsize=fig_size)  # Adjust the figure size here

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)  # Get the class name
            plt.imshow(image)  # Display the image
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=1, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')

    plt.axis('off')  # Turn off axis
    plt.show()


# Display the image with bounding boxes and correct labels
draw_boxes(Image.open(image_path), prediction, fig_size=(9, 16))  # Example of increased size
