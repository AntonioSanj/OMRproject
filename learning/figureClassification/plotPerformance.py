import json
import matplotlib.pyplot as plt
from constants import *

with open(figuresPerformanceDataJson, "r") as f:
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
plt.ylim(0, 1)
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.show()
