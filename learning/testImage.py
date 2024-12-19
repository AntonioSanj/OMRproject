import cv2
import torch
import torchvision.transforms as t

from utils.plotUtils import showImage


def testImage(path, model, device):

    original_image = cv2.imread(path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with model input

    # Preprocess the image (resize, normalize, and convert to tensor)
    transform = t.Compose([
        t.ToTensor(),  # Convert the image to a tensor
    ])
    input_image = transform(image_rgb).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Get predictions
    with torch.no_grad():
        predictions = model(input_image)
    print(predictions)

    # Postprocess predictions
    threshold = 0.5  # Confidence threshold
    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    pred_labels = predictions[0]['labels']

    # Filter boxes based on the threshold
    filtered_boxes = pred_boxes[pred_scores > threshold].cpu().numpy()
    filtered_scores = pred_scores[pred_scores > threshold].cpu().numpy()
    filtered_labels = pred_labels[pred_scores > threshold].cpu().numpy()

    # Draw bounding boxes on the original image
    for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
        xmin, ymin, xmax, ymax = map(int, box)
        # Draw the rectangle
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box
        # Add the label and score
        label_text = f"Class {label}: {score:.2f}"
        cv2.putText(original_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    showImage(original_image, "predictions")