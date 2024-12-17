import torch
import torchvision.ops as ops


def compute_iou(pred_boxes, target_boxes):
    # intersection over union
    iou = ops.box_iou(pred_boxes, target_boxes)
    return iou


def calculate_accuracy(predictions, targets, iou_threshold=0.5):
    """
    Calculate accuracy metrics for object detection.
    - Precision and recall based on IoU threshold.
    """
    num_true_positives = 0
    num_predicted_boxes = 0
    num_ground_truth_boxes = 0

    for prediction, target in zip(predictions, targets):
        pred_boxes = prediction['boxes']
        pred_labels = prediction['labels']
        target_boxes = target['boxes']
        target_labels = target['labels']

        # Count ground truth boxes
        num_ground_truth_boxes += len(target_boxes)

        # Compute IoU between predicted and target boxes
        ious = compute_iou(pred_boxes, target_boxes)

        # For each predicted box, check if it matches a ground truth box
        for iou in ious:
            if torch.max(iou) >= iou_threshold:  # If IoU > threshold, consider it a match
                num_true_positives += 1
                num_predicted_boxes += 1
            else:
                num_predicted_boxes += 1

    # Precision and Recall
    precision = num_true_positives / num_predicted_boxes if num_predicted_boxes > 0 else 0
    recall = num_true_positives / num_ground_truth_boxes if num_ground_truth_boxes > 0 else 0

    return precision, recall
