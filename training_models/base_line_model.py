import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = os.environ.get("ASL_TRAIN_DIR")

if not data_dir:
    raise RuntimeError(
        "ASL_TRAIN_DIR is not defined."
    )

if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"Training dataset was not found: {data_dir}"
    )

print("Training dataset:", data_dir)
# 1. Base settings

batch_size = 64
epochs = 10
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "Enable GPU in Colab before running this script."
    )

device = torch.device("cuda")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch CUDA version:", torch.version.cuda)

# GPU -> baseline, CPU -> baseline_cpu_run_1
run_name = "baseline"
# Docker: /results is connected to the project's results folder.
# Local run: save inside ./results as well.
drive_output_dir = os.environ.get("ASL_OUTPUT_DIR")

if drive_output_dir:
    OUTPUT_DIR = os.path.join(drive_output_dir, run_name)
elif os.path.exists("/results"):
    OUTPUT_DIR = os.path.join("/results", run_name)
else:
    OUTPUT_DIR = PROJECT_ROOT / "results" / run_name

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Results directory:", OUTPUT_DIR)



best_val_acc = 0.0
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_baseline_model.pth")

# 2. Data preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,     pin_memory=(device.type == "cuda"))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),# Flatten: [64 channels * 16 height * 16 width] -> 16,384 features
                                         # Latent space: Compressed to 128 dimensions for classification
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 4. Setup
model = SimpleCNN(num_classes=len(full_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# 5. Training loop
print(f"Starting training on {device}...")

start_time = time.time()

for epoch in range(epochs):
    print(f"--- Starting Epoch {epoch+1}/{epochs} ---")

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

    print(f"Epoch {epoch+1} Summary | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]:.2f}%")

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

print(f"Training finished! Results plot saved to {OUTPUT_DIR}")
print(f"Best model saved to {BEST_MODEL_PATH}")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Training Time: {minutes}m {seconds}s")
