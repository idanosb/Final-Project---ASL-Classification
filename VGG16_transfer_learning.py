import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time


if os.path.exists('/data/asl_alphabet_train'):
    data_dir = '/data/asl_alphabet_train'
else:
    data_dir = r'D:\idan\ASLData\asl_alphabet_train'

# 1. Base settings
batch_size = 64
epochs = 10
unfreeze_epoch = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch CUDA version:", torch.version.cuda)

OUTPUT_DIR = "/results/VGG16_transfer_learning" if os.path.exists("/results") else "./VGG16_transfer_learning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

best_val_acc = 0.0
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "VGG16_transfer_learning.pth")

# 2. Data preparation
weights = VGG16_Weights.DEFAULT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Model architecture - VGG16 Transfer Learning
num_classes = len(full_dataset.classes)

model = vgg16(weights=weights)

# Freeze all convolutional feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classifier layer for ASL classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

# 4. Setup
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# 5. Training loop
print(f"Starting VGG16 fine-tuning on {device}...")

start_time = time.time()

for epoch in range(epochs):
    print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

    if epoch == unfreeze_epoch:
        print("Unfreezing last VGG16 convolution block...")

        for param in model.features[24:].parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001
        )

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct_train / total_train)

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(100 * correct_val / total_val)

    if val_accuracies[-1] > best_val_acc:
        best_val_acc = val_accuracies[-1]
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

    print(
        f"Epoch {epoch + 1} Summary | "
        f"Train Loss: {train_losses[-1]:.4f} | "
        f"Val Loss: {val_losses[-1]:.4f} | "
        f"Val Acc: {val_accuracies[-1]:.2f}%"
    )

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

# 6. Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='purple')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.title('Loss Evolution')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='purple')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Accuracy Evolution')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.savefig(os.path.join(OUTPUT_DIR, "training_results.png"))

summary_path = os.path.join(OUTPUT_DIR, "training_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Training Time: {minutes}m {seconds}s\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Best Model Path: {BEST_MODEL_PATH}\n")
    f.write("Model: VGG16\n")
    f.write("Transfer Learning: Yes\n")
    f.write("Frozen layers: VGG16 features initially frozen\n")
    f.write(f"Unfreeze epoch: {unfreeze_epoch + 1}\n")
    f.write("Unfrozen layers: model.features[24:]\n")

print(f"Training finished! Results plot saved to {OUTPUT_DIR}")
print(f"Best model saved to {BEST_MODEL_PATH}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Training Time: {minutes}m {seconds}s")