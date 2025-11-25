import torch
from cnn_train import CNN
import torch.nn as nn
from image_processing import test_loader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = CNN()                  
model.load_state_dict(torch.load("models/model7.pth", map_location=device))
model.to(device)

model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum= 0.9, weight_decay=0.0005)

model.eval()
val_loss = 0
correct = 0
total = 0
batch = 0

if __name__ == "__main__":
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            val_loss += loss.item() * images.size(0)
            batch += 1
            print(f"Batch {batch} completed")
        val_loss = val_loss/batch
    print("Accuracy: ", correct/total)
    print("Val_loss: ", val_loss)

