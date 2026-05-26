import tempfile
import shutil

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import AlexNet


device = "cuda" if torch.cuda.is_available() else "cpu"


temp_dir = tempfile.mkdtemp()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_dataset = datasets.CIFAR10(
    root=temp_dir,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=temp_dir,
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

model = AlexNet(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 10

for epoch in range(epochs):
    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")



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

print(f"Accuracy: {accuracy:.2f}%")



shutil.rmtree(temp_dir)
