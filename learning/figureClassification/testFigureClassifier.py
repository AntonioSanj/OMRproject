import os

import torch.nn as nn
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

from constants import *


def startClassModel(model_path, num_classes=9):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def classifyFigure(image_path, model):
    # Image preprocessing
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Define class names (Ensure these match your training data)
    class_names = ['double', 'fClef', 'four', 'gClef', 'half', 'one', 'quarter', 'restHalf', 'restOne']

    predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name


def classify_all_figures(dataset_path, model):
    class_correct = {}
    class_total = {}
    total_correct = 0
    total_images = 0
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            class_correct[category] = 0
            class_total[category] = 0

            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                if os.path.isfile(image_path) and image_file.endswith(
                        ('.png', '.jpg', '.jpeg')):  # Check for valid images
                    prediction = classifyFigure(image_path, model)
                    class_total[category] += 1
                    total_images += 1

                    if prediction.lower() == category.lower():  # Case-insensitive comparison
                        class_correct[category] += 1
                        total_correct += 1
                    else:
                        print(f'{category} was classified as {prediction}')

    # Print class-wise accuracy
    print("Accuracy per class:")
    for category in class_correct:
        accuracy = (class_correct[category] / class_total[category]) * 100 if class_total[category] > 0 else 0
        print(f'{category.upper()}: {accuracy:.2f}% ({class_correct[category]}/{class_total[category]})')

    # print overall accuracy
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_images})")
    print("")


def testFigureClassification(model_dir, image_dir):
    model = startClassModel(model_dir)
    classify_all_figures(image_dir, model)


modelFigureClass = startClassModel(figureModels + 'figure_classification_model.pth')

print("\nTRAINING SET")
# classify_all_figures(myFiguresDataSet, modelFigureClass)

print("TEST SET")
classify_all_figures(myFiguresDataSetTest, modelFigureClass)
