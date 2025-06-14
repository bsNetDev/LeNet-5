import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import torchvision
from utils import compute_mean_std
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# LeNet-5 model definition
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)  # 50% drop rate

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)


raw_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
mean, std = compute_mean_std(raw_dataset)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/lenet5_CIFAR10')

images, labels = next(iter(train_loader))
img_grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.transpose(img_grid.cpu().numpy(), (1, 2, 0)))
ax.axis('off')
for idx in range(16):
    row = idx // 4
    col = idx % 4
    label = labels[idx].item()
    ax.text(col * 32 + 2, row * 32 + 10, str(label), color='yellow', fontsize=8, backgroundcolor='black')

writer.add_figure('CIFAR10/Train_Images_With_Labels', fig, global_step=0)
plt.close(fig)

# Tracking
train_losses = []
test_accuracies = []

# Training
for epoch in range(50):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch == 0 and batch_idx == 0:
            probabilities = torch.nn.functional.softmax(output, dim=1)  # shape [batch, 10]
            writer.add_histogram('Distributions/Train_Softmax_E0_B0', probabilities, epoch)


    train_losses.append(running_loss)
    writer.add_scalar("Loss/train", running_loss, epoch)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = correct / total
    test_accuracies.append(accuracy)
    writer.add_scalar("Accuracy/test", accuracy, epoch)
    print(f"Test Accuracy: {accuracy:.4f}")

    if epoch == 4:
        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        class_names = train_dataset.classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        writer.add_figure("ConfusionMatrix/Final_Epoch", fig, global_step=epoch)
        plt.close(fig)
    scheduler.step()
writer.close()

# Plotting
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, marker='o', color='green')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/training_metrics.png")

torch.save(model.state_dict(), "lenet5_cifar10.pth")
