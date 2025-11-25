import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from cnn_train import CNN   # your model file

# ------------------------------
# Path Setup
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "dogs-vs-cats", "test1")
print("Base Directory:", BASE_DIR)
print("Test Images Path:", img_path)

# ------------------------------
# Load Model
# ------------------------------
device = torch.device("cpu")

model = CNN()
model.load_state_dict(torch.load("models/model7.pth", map_location=device))
model.to(device)
model.eval()

# ------------------------------
# Visualize First Convolution Layer
# ------------------------------
print("Visualizing conv1 filters...")

first_conv = model.conv1.weight.data.cpu()

fig, axarr = plt.subplots(4, 8, figsize=(12, 6))
for idx, ax in enumerate(axarr.flatten()):
    if idx >= first_conv.shape[0]:
        break

    # conv1 has shape: (16 filters, 3 channels, H, W)
    ax.imshow(first_conv[idx].permute(1, 2, 0))  # CHW → HWC
    ax.axis("off")

plt.suptitle("Conv1 Filters")
# plt.show()

# ------------------------------
# Visualize Second Convolution Layer
# ------------------------------
print("Visualizing conv2 filters...")

second_conv = model.conv2.weight.data.cpu()

fig, axarr = plt.subplots(4, 8, figsize=(12, 6))
for idx, ax in enumerate(axarr.flatten()):
    if idx >= second_conv.shape[0]:
        break

    # conv2 has 32 filters, each 16 channels. Too many channels → average to grayscale
    filt = second_conv[idx].mean(dim=0)

    ax.imshow(filt, cmap="gray")
    ax.axis("off")

plt.suptitle("Conv2 Filters (Averaged Channels)")
# plt.show()

fourth_conv = model.conv4.weight.data.cpu()

fig, axarr = plt.subplots(4, 8, figsize=(12, 6))
for idx, ax in enumerate(axarr.flatten()):
    if idx >= second_conv.shape[0]:
        break

    # conv2 has 32 filters, each 16 channels. Too many channels → average to grayscale
    filt = fourth_conv[idx].mean(dim=0)

    ax.imshow(filt, cmap="gray")
    ax.axis("off")

plt.suptitle("Conv3 Filters (Averaged Channels)")
plt.show()
