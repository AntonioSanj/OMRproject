import json

import matplotlib.pyplot as plt

from constants import *

file = figuresPerformanceDataJson2

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
