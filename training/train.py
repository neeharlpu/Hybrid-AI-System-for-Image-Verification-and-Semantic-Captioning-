
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time

# =========================
# DEVICE
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# =========================
# CONFIG
# =========================
DATA_DIR = "../dataset"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001
MODEL_PATH = "../models/model.pth"

# =========================
# TRANSFORMS (IMPROVED)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# LOAD DATA
# =========================
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# 🔥 Unfreeze deeper layers (IMPORTANT)
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i, (images, labels) in enumerate(train_loader):

        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s")

# =========================
# EVALUATION
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
