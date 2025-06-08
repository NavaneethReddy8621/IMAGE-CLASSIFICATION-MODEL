import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_size=int(0.8 * len(dataset))
val_size=len(dataset) - train_size
train_dataset,val_dataset=random_split(dataset, [train_size, val_size])
train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN,self).__init__()
        self.conv_block=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32, 64, 3, padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.dropout = nn.Dropout(0.3)
        self.fc_block = nn.Sequential(nn.Linear(128 * 4 * 4, 256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256, 10))
    def forward(self, x):
        x=self.conv_block(x)
        x=torch.flatten(x, 1)
        x=self.fc_block(x)
        return x
cnn_model=CustomCNN().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(cnn_model.parameters(), lr=0.001)
train_losses,val_losses,train_accuracies,val_accuracies=[],[],[],[]
for epoch in range(15):
    cnn_model.train()
    train_loss,correct_train,total_train=0.0,0,0
    for inputs,targets in train_loader:
        inputs,targets=inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs=cnn_model(inputs)
        loss=loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        _, predicted=torch.max(outputs, 1)
        total_train+=targets.size(0)
        correct_train+=(predicted==targets).sum().item()
    cnn_model.eval()
    val_loss,correct_val,total_val=0.0,0,0
    with torch.no_grad():
        for inputs,targets in val_loader:
            inputs,targets=inputs.to(device),targets.to(device)
            outputs=cnn_model(inputs)
            loss=loss_fn(outputs,targets)
            val_loss+=loss.item()
            _, predicted=torch.max(outputs,1)
            total_val+=targets.size(0)
            correct_val+=(predicted==targets).sum().item()

    train_accuracy=100*correct_train/total_train
    val_accuracy=100 *correct_val/total_val
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
cnn_model.eval()
all_preds,all_labels=[],[]
correct_test,total_test=0,0
with torch.no_grad():
    for inputs,targets in test_loader:
        inputs,targets=inputs.to(device), targets.to(device)
        outputs=cnn_model(inputs)
        _, predicted=torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        correct_test+=(predicted == targets).sum().item()
        total_test+=targets.size(0)

test_accuracy=100*correct_test/total_test
print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print(classification_report(all_labels,all_preds,target_names=test_dataset.classes))
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(train_losses,label='Train Loss')
plt.plot(val_losses,label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

torch.save(cnn_model.state_dict(), 'enhanced_cifar10_cnn.pth')
