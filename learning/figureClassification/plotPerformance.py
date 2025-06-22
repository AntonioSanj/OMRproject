import json

import matplotlib.pyplot as plt

from constants import *


def classificationLatex2(file):
    with open(file, "r") as f:
        data = json.load(f)

    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    num_epochs = len(train_accuracies)

    # Start LaTeX table
    latex = "\\begin{table}[H]\n\\centering\n\\small\n"
    latex += "\\setlength{\\tabcolsep}{6pt}\n\\renewcommand{\\arraystretch}{1.1}\n"
    latex += "\\begin{tabular}{c cc cc}\n"
    latex += "\\toprule\n"

    # First header row
    latex += "& \\multicolumn{2}{c}{Loss} & \\multicolumn{2}{c}{Accuracy (\\%)} \\\\\n"
    latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"

    # Second header row
    latex += "Epoch & Train & Val & Train & Val \\\\\n"
    latex += "\\midrule\n"

    # Rows for each epoch
    for epoch in range(num_epochs):
        row = [
            str(epoch + 1),
            f"{train_losses[epoch]:.4f}",
            f"{val_losses[epoch]:.4f}",
            f"{train_accuracies[epoch]*100:.2f}",
            f"{val_accuracies[epoch]*100:.2f}"
        ]
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n\\end{table}"

    return latex


def plotClassification(file):
    with open(file, "r") as f:
        data = json.load(f)

    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    epochs = range(1, len(train_accuracies) + 1)

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.savefig(figClassPerformanceDataDir + "/accuracy_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 6)
    plt.legend()
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.savefig(figClassPerformanceDataDir + "/loss_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


plotClassification(figuresPerformanceDataJson2)
print(classificationLatex2(figuresPerformanceDataJson2))
