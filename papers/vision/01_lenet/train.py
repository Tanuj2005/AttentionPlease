import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root="./temp_data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./temp_data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


model = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr= 1e-3
)

epochs = 5

for epoch in range( epochs ):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion( outputs, labels )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")


model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

shutil.rmtree("./temp_data")