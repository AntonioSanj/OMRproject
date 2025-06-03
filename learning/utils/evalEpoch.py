import torch
from matplotlib import pyplot as plt, patches
import torchvision.transforms.functional as F


def evaluate_one_epoch(model, data_loader, device, coco_gt, score_thresh=0.15, iou_thresh=0.5):
    model.eval()

    total_preds = 0
    total_gt = 0
    correct_localization = 0
    correct_classification = 0
    iou_scores = []

    categories = {v['id']: v['name'] for v in coco_gt.loadCats(coco_gt.getCatIds())}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for image_tensor, target, output in zip(images, targets, outputs):
                image_id = int(target[0]['image_id']) if isinstance(target, list) else int(target['image_id'])

                # evaluation is only done in those that are given a high enough score
                scores = output['scores'].cpu()
                keep_idxs = scores >= score_thresh
                boxes = output["boxes"].cpu()[keep_idxs]
                labels = output["labels"].cpu()[keep_idxs]

                gt_boxes, gt_labels = extract_ground_truth(coco_gt, image_id)

                visualize_preds_vs_groundtruth(image_tensor, output, gt_boxes, categories, score_thresh=score_thresh)

                total_preds += len(boxes)
                total_gt += len(gt_boxes)

                for pred_box, pred_label in zip(boxes, labels):
                    best_iou = 0.0
                    best_idx = -1

                    for j, gt_box in enumerate(gt_boxes):
                        iou_val = IoU(pred_box.tolist(), gt_box.tolist())
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_idx = j

                    if best_iou >= iou_thresh:
                        correct_localization += 1
                        if pred_label == gt_labels[best_idx]:
                            correct_classification += 1

                    iou_scores.append(best_iou)

    loc_acc = (correct_localization / total_preds) * 100 if total_preds > 0 else 0.0
    cls_acc = (correct_classification / correct_localization) * 100 if correct_localization > 0 else 0.0
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    print(f"Segmentation Accuracy: {loc_acc:.2f}%")
    print(f"Classification Accuracy: {cls_acc:.2f}%")
    print(f"Average IoU: {avg_iou:.2f}")


def extract_ground_truth(coco, image_id):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    gt_boxes = []
    gt_labels = []

    for ann in anns:
        if ann.get("iscrowd", 0):
            continue  # skip crowd annotations
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(ann['category_id'])

    if len(gt_boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,), dtype=torch.int64)

    return torch.tensor(gt_boxes), torch.tensor(gt_labels)


def visualize_preds_vs_groundtruth(image_tensor, output, gt_boxes, categories, score_thresh=0.15):
    image = F.to_pil_image(image_tensor.cpu())
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Ground Truth boxes (green)
    for box in gt_boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    # Predicted boxes (red)
    boxes = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        best_iou = 0.0
        for gt_box in gt_boxes:
            iou = IoU(box.tolist(), gt_box.tolist())
            best_iou = max(best_iou, iou)

        label_name = categories.get(int(label), str(label))
        color = 'blue' if best_iou > 0.5 else 'pink'
        ax.text(x1, y2 + 5, f'{best_iou:.2f}', color=color, fontsize=10)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0
