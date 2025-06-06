import json
import os

from matplotlib import pyplot as plt

from constants import frcnnPerformanceDataDir


def scoreFromFile(filename):
    # Example: 'performanceData_0_15.json' -> 0.15
    base = os.path.splitext(filename)[0]  # remove .json
    parts = base.split('_')               # split by underscore
    return float(f"{parts[-2]}.{parts[-1]}")


def plot_metrics(fullpath, score=None):
    with open(fullpath, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(12, 8))

    colors = {
        "precision": "blue",
        "recall": "green",
        "f1_score": "red",
        "classification_accuracy": "yellow",
        "average_iou": "purple"
    }

    for idx, (metric, values) in enumerate(data.items()):
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
        fontsize=14,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),  # (x, y) in axes fraction: near bottom-right inside
        borderaxespad=0,
        frameon=True
    )
    plt.grid(True)
    plt.xticks(range(1, 21))
    plt.tight_layout()
    plt.savefig(f"fullScore_{score}".replace('.', '_') + '.png')
    plt.show()


def plotAllMetricDataFiles(path):
    for dataFile in os.listdir(path):
        if dataFile.endswith('.json'):
            fullPath = os.path.join(path, dataFile)
            scoreThresh = scoreFromFile(dataFile)
            plot_metrics(fullPath, score=scoreThresh)


plotAllMetricDataFiles(frcnnPerformanceDataDir + 'full')
