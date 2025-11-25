from matplotlib import pyplot as plt
from image_processing import train_loader, test_loader
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)    
        self.conv3 = nn.Conv2d(32,64,3, padding=1)    
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128 *14 * 14, 256)
        self.dropout = nn.Dropout(p=0.3)  # 30% of neurons dropped during training
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = self.pool(F.relu(self.conv3(out)))
        out = self.pool(F.relu(self.conv4(out)))

        out = out.view(out.size(0), -1) # Flattens it into dimensions automatically computed by pytorch, using batch size (x.size(0))

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    

if __name__ == "__main__": #Need this on mac to use num_workers

    #Inicialization Shenanigans 
    model = CNN()
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum= 0.9, weight_decay=0.0005)
    
    #For graphs
    train_losses = []
    val_losses = []
    
    for epoch in range(25):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")
            
            running_loss += loss.item() * images.size(0)  # sum over batch

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * images.size(0)
        epoch_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")


    torch.save(model.state_dict(), "model5.pth")
    
    print(f"train_losses: {train_losses[-5:]}, test_losses: {val_losses[-5:]}")
    plt.plot(range(1, 26), train_losses, label="Train Loss")
    plt.plot(range(1, 26), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss vs Epoch")
    plt.legend()
    plt.show()
    
    
    
# model1.pth -- first attempt, lr = 0.001, no momentum. Running on CPU. Ending loss around 0.58
# model2.pth -- second attempt, lr = 0.01, momentum = 0.09, switched to running on mps. Ending loss slightly lower, around 0.48
#model3.pth -- third attempt. lr = 0.01, momentum = 0.9. Increased from 3 convolutional and pooling layers to 4. Increased from 5 epochs to 10. 
#      Signifigant improvement with loss ranging from 0.07-0.3 normally.

##On this attempt I comment out save model.pth, since this is a kaggle dataset I will run again but with train/test split
#Note this is because kaggle datasets are unlabled, therefore I am unable to compute accuracy/loss with them. This attempt
#will use the same architecture as model3.pth, only that the data it is trained on will only be 80% of when I trained it 
#to produce model3.pth. The produced output: Epoch 10, Val Loss: 0.2665, Val Accuracy: 0.8930


#model4.pth -- Increased batches to size 48, 15 epochs. final epoch training loss:0.1463615158125758

#model4.pth -- Alright, time for the big guns. Lets train on 50 epochs 
# Epoch 28, Train Loss: 0.0472, Val Loss: 0.3059 -- clear overfitting so I stopped it. 

#model4.pth (again) -- since the last model5 never finished, it was never saved. I got rid of the 4th convolutional layer I added earlier,
#added 0.3 dropout in between fc1 and fc2, as well as added a decay weight of 0.0001 to the optimizer
#Still overfitting, lets try again

##model5.pth, 25 epochs Epoch 25, Train Loss: 0.1843, Val Loss: 0.3006

##model6.pth, reduce rotation to 7.5 Epoch 25, Train Loss: 0.0876, Val Loss: 0.3924
#one of the worst ones yet. 

#model7.pth -- re added 4th conv layer and put rotation to 10. Best results by far. Epoch 25, Train Loss: 0.0831, Val Loss: 0.2234.
# and Accuracy:  0.9694 ##NOTE all of these val_losses have been averages during training. Once using only the trained model, val_loss =7.914553542506127 
