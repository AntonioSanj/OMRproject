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

inverse_train_acc = [1 - acc for acc in train_accuracies]
inverse_val_acc = [1 - acc for acc in val_accuracies]

epochs = range(1, len(train_accuracies) + 1)

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, inverse_train_acc, label="Training Accuracy")
plt.plot(epochs, inverse_val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.yscale("log")
plt.ylim(0.01, 1)
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0.2, 0.6)
plt.legend()
plt.show()
