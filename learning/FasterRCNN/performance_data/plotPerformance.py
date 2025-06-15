import json
import os

from matplotlib import pyplot as plt

from constants import frcnnPerformanceFull, frcnnPerformanceSlice


def scoreFromFile(filename):
    # Example: 'performanceData_0_15.json' -> 0.15
    base = os.path.splitext(filename)[0]  # remove .json
    parts = base.split('_')  # split by underscore
    return float(f"{parts[-2]}.{parts[-1]}")


def plot_metrics(fullpath, score=None, imageName=None):
    with open(fullpath, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(12, 8))

    colors = {
        "precision": "blue",
        "recall": "green",
        "f1_score": "red",
        "classification_accuracy": "yellow",
        "IoU": "purple",
        "iou": "purple",
        "average_iou": "purple"
    }

    for idx, (metric, values) in enumerate(data.items()):
        if metric != 'classification_accuracy':
            metric = 'iou' if metric == 'average_iou' else metric
            color = colors.get(metric, 'black')
            plt.plot(
                range(1, len(values) + 1),
                values,
                label=f"{metric.replace('_', ' ').title()}",
                linewidth=2.5,
                color=color
            )

    plt.title(f'Score Threshold {score}', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Metric Value (%)', fontsize=16)
    plt.yticks(range(0, 101, 10))
    plt.legend(
        fontsize=18,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),  # (x, y) in axes fraction: near bottom-right inside
        borderaxespad=0,
        frameon=True
    )
    plt.grid(True)
    plt.xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig(f"{imageName}_{score}".replace('.', '_') + '.png') if imageName else None
    plt.show()


def plotClassification(folderDir, title='', save=None):
    plt.figure(figsize=(12, 8))
    colors = ["blue",
              "green",
              "red",
              "yellow",
              "purple",
              "pink"]
    color_i = 0
    for dataFile in os.listdir(folderDir):
        if dataFile.endswith('.json'):
            scoreThresh = scoreFromFile(dataFile)

            with open(os.path.join(folderDir, dataFile), 'r') as f:
                data = json.load(f)

            acc = data.get('classification_accuracy')
            if acc is None or len(acc) != 20:
                print(f"Skipping {dataFile}: Invalid or missing 'classification_accuracy'")
                continue
            color = colors[color_i % len(colors)]
            epochs = range(1, 21)
            plt.plot(epochs, acc, label=f"Threshold = {scoreThresh}", color=color)
            color_i += 1

    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Classification Accuracy (%)', fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.xticks(range(1, 21))
    plt.yticks(range(0, 101, 10))
    plt.legend(
        fontsize=18,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),  # (x, y) in axes fraction: near bottom-right inside
        borderaxespad=0,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(save + '.png') if save else None
    plt.show()


def plotAllMetricDataFiles(path, imageName=None):
    for dataFile in os.listdir(path):
        if dataFile.endswith('.json'):
            fullPath = os.path.join(path, dataFile)
            scoreThresh = scoreFromFile(dataFile)
            plot_metrics(fullPath, score=scoreThresh, imageName=imageName)


plotAllMetricDataFiles(frcnnPerformanceSlice + '/train', 'train_sliced')
plotAllMetricDataFiles(frcnnPerformanceSlice + '/val', 'val_sliced')
# plotClassification(frcnnPerformanceSlice + '/train', 'Training set classification accuracy', save='trainsliceclass')
# plotClassification(frcnnPerformanceSlice + '/val', 'Validation set classification accuracy', save='valsliceclass')
