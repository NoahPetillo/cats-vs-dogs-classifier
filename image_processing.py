import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

def imshow(img_tensor):
    img = img_tensor.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean    # undo normalization
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")

class TestDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths.sort() 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Extract label from filename
        # Assumes filenames like 'cat.123.jpg' or 'dog.456.jpg'
        filename = os.path.basename(img_path).lower()
        if "cat" in filename:
            label = 0
        elif "dog" in filename:
            label = 1
        else:
            raise ValueError(f"Unknown label in file {filename}")

        return img, label




#Inicialize transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize all images
    transforms.RandomHorizontalFlip(),           # Augmentation
    transforms.RandomRotation(10),               # Augmentation
    transforms.ToTensor(),                       # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization -- averages of each RGB pixel
                         std=[0.229, 0.224, 0.225]),
])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225])
])

#Need this part to work with train and test loader as imports in CNN.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dogs-vs-cats", "train")
TEST_DIR  = os.path.join(BASE_DIR, "dogs-vs-cats", "test1")

#Load data
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_data  = TestDataset(TEST_DIR, transform=test_transforms)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, test_dataset = random_split(train_data, [train_size, val_size])


#Create data loader
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
test_loader   = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=4)

# images, labels = next(iter(train_loader))
# print(images.shape)   # torch.Size([32, 3, 224, 224])
# print(labels[:10])

# for i in range(16):
#     plt.subplot(4, 4, i + 1)

#     imshow(images[i])
# plt.show()

