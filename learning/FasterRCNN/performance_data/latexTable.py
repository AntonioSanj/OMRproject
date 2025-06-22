import json
import os

from learning.FasterRCNN.performance_data.plotPerformance import scoreFromFile


def json_to_latex_table(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    data.pop('classification_accuracy', None)
    epochs = len(next(iter(data.values()), []))

    headers = ["Epoch"] + list(data.keys())
    headers = [head if head == 'IoU' else head.replace("_", " ").title() for head in headers]

    # Begin LaTeX table with booktabs
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
    latex += "\\begin{tabular}{" + "c" * len(headers) + "}\n"
    latex += "\\toprule\n"
    latex += " & ".join(headers) + " (\\%) \\\\\n"
    latex += "\\midrule\n"

    for epoch in range(epochs):
        row = [str(epoch + 1)]
        for metric in data.keys():
            row.append(f"{data[metric][epoch]:.2f}")
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n\\end{table}"
    return latex


def latexClassification(folderDir):
    threshold_accuracies = {}

    for dataFile in os.listdir(folderDir):
        if dataFile.endswith('.json'):
            scoreThresh = scoreFromFile(dataFile)

            with open(os.path.join(folderDir, dataFile), 'r') as f:
                data = json.load(f)

            acc = data.get('classification_accuracy')
            if acc is None or len(acc) != 20:
                return 'error'

            threshold_accuracies[scoreThresh] = acc

    # Sort thresholds numerically for consistent table order
    sorted_thresholds = sorted(threshold_accuracies.keys())

    # Start LaTeX table
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += "\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
    latex += "\\begin{tabular}{" + "c" * (len(sorted_thresholds) + 1) + "}\n"
    latex += "\\toprule\n"

    # Header row
    headers = ["Epoch"] + [f"{thresh}" for thresh in sorted_thresholds]
    latex += " & ".join(headers) + " (\\%) \\\\\n"
    latex += "\\midrule\n"

    # Table rows per epoch
    num_epochs = 20
    for epoch in range(num_epochs):
        row = [str(epoch + 1)]
        for thresh in sorted_thresholds:
            row.append(f"{threshold_accuracies[thresh][epoch]:.2f}")
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n\\end{table}"

    return latex


path = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/performance_data/sliced/val/1_performance_0_30.json'
path2 = 'C:/Users/Usuario/Desktop/UDC/QUINTO/TFG/src_code/learning/FasterRCNN/performance_data/full/2/val'

# print(json_to_latex_table(path))
print(latexClassification(path2))
