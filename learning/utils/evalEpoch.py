import torch
from matplotlib import pyplot as plt, patches
from torchvision.ops import box_iou


def evaluate_one_epoch(model, data_loader, device, coco_gt, score_thresh=0.05, iou_thresh=0.7):
    model.eval()

    total_preds = 0
    total_gt = 0
    correct_localization = 0
    correct_classification = 0
    iou_scores = []

    categories = {v['id']: v['name'] for v in coco_gt.loadCats(coco_gt.getCatIds())}
    valid_ids = set(coco_gt.getImgIds())

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            if batch_idx == 0:
                visualize_predictions(images[0], outputs[0], categories, score_thresh=0.15)

            for target, output in zip(targets, outputs):
                image_id = int(target[0]['image_id']) if isinstance(target, list) else int(target['image_id'])

                if image_id not in valid_ids:
                    print(f"[WARNING] image_id {image_id} not found in COCO ground truth!")
                    continue

                # evaluation is only done in those that are given a high enough score
                scores = output['scores'].cpu()
                keep_idxs = scores >= score_thresh
                boxes = output["boxes"].cpu()[keep_idxs]
                labels = output["labels"].cpu()[keep_idxs]

                gt_boxes, gt_labels = extract_ground_truth(coco_gt, image_id)
                total_preds += len(boxes)
                total_gt += len(gt_boxes)

                ious = box_iou(boxes, gt_boxes)

                for i, (box, label) in enumerate(zip(boxes, labels)):
                    # select best IoU of the predicted box
                    iou_vals = ious[i]
                    best_idx = torch.argmax(iou_vals).item()
                    best_iou = iou_vals[best_idx].item()

                    # if it matches enough with any GT segmentation is good
                    if best_iou >= iou_thresh:
                        correct_localization += 1
                        if label == gt_labels[best_idx]:
                            correct_classification += 1

                    iou_scores.append(best_iou)
    loc_acc = (correct_localization / total_preds)*100 if total_preds > 0 else 0.0
    cls_acc = (correct_classification / correct_localization)*100 if correct_localization > 0 else 0.0
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


def visualize_predictions(image_tensor, predictions, categories, score_thresh=0.15):
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    if image.max() > 1:
        image = image / 255.0

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{categories[label]}: {score:.2f}",
                color='red', fontsize=12)

    plt.axis('off')
    plt.show()
