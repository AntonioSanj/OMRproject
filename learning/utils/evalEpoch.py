import json
import os

import torch
from matplotlib import pyplot as plt, patches
from pycocotools.cocoeval import COCOeval


def evaluate_one_epoch(model, data_loader, device, epoch, coco_gt):
    model.roi_heads.score_thresh = 0.001
    model.eval()

    results = []
    categories = {v['id']: v['name'] for v in coco_gt.loadCats(coco_gt.getCatIds())}
    valid_ids = set(coco_gt.getImgIds())
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)

            if batch_idx == 0:
                visualize_predictions(images[0], outputs[0], categories, score_thresh=0.15)
            for target, output in zip(targets, outputs):
                if isinstance(target, list):
                    image_id = int(target[0]['image_id'])
                else:
                    image_id = int(target['image_id'])

                if image_id not in valid_ids:
                    print(f"[Warning] image_id {image_id} not found in COCO ground truth!")
                    continue
                image_id = target[0]['image_id'] if isinstance(target, list) else target['image_id']
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    results.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],
                        "score": float(score),
                    })

    # Save predictions to JSON (temporary file)
    pred_file = f"temp_predictions_epoch_{epoch}.json"
    with open(pred_file, "w") as f:
        json.dump(results, f, indent=4)

    coco_dt = coco_gt.loadRes(pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Optional: remove temp file
    os.remove(pred_file)


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
