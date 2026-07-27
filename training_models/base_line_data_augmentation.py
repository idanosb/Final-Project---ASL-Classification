import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. Project paths and dataset validation
# ============================================================

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


# ============================================================
# 2. Base experiment settings
# ============================================================

batch_size = 64
epochs = 10
learning_rate = 0.001

# Using a fixed seed makes the train-validation split reproducible.
random_seed = 42

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "Enable GPU in Colab before running this script."
    )

device = torch.device("cuda")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Torch CUDA version:", torch.version.cuda)


# A separate run name prevents overwriting the original baseline results.
run_name = "baseline_data_augmentation"

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

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_baseline_data_augmentation_model.pth"
)


# ============================================================
# 3. Data transformations
# ============================================================

# Data augmentation is applied only to the training set.
#
# The transformations are intentionally moderate:
# - Small rotations simulate changes in hand angle.
# - Translation simulates different hand positions in the image.
# - Scaling simulates the hand being closer to or farther from the camera.
# - Brightness and contrast changes simulate different lighting conditions.
#
# Horizontal flipping is not used because flipping an ASL hand sign may
# create an unrealistic or semantically different training example.

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10)
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Validation images must not receive random augmentation.
# They only receive the same deterministic preprocessing used
# in the original baseline experiment.

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 4. Create a reproducible train-validation split
# ============================================================

# This dataset is used only to determine the class names,
# dataset length and split indices. No transformation is needed here.
index_dataset = datasets.ImageFolder(
    data_dir
)

train_size = int(0.8 * len(index_dataset))
val_size = len(index_dataset) - train_size

split_generator = torch.Generator().manual_seed(random_seed)

train_split, val_split = random_split(
    index_dataset,
    [train_size, val_size],
    generator=split_generator
)


# Create two separate ImageFolder datasets so that the training set
# and validation set can use different transformations.
train_full_dataset = datasets.ImageFolder(
    data_dir,
    transform=train_transform
)

val_full_dataset = datasets.ImageFolder(
    data_dir,
    transform=val_transform
)


# Use exactly the same indices created by the reproducible split.
train_dataset = Subset(
    train_full_dataset,
    train_split.indices
)

val_dataset = Subset(
    val_full_dataset,
    val_split.indices
)

print("Number of classes:", len(index_dataset.classes))
print("Class names:", index_dataset.classes)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# 5. Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# 6. Baseline CNN architecture
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # After two pooling layers:
            # 64x64 -> 32x32 -> 16x16
            #
            # Flattened feature size:
            # 64 channels * 16 height * 16 width = 16,384 features
            nn.Linear(
                64 * 16 * 16,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 7. Model, loss function and optimizer
# ============================================================

model = SimpleCNN(
    num_classes=len(index_dataset.classes)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# 8. Training and validation loop
# ============================================================

print(f"Starting training on {device}...")

start_time = time.time()

for epoch in range(epochs):
    print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---")

    # --------------------------------------------------------
    # Training phase
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward propagation.
        outputs = model(images)

        # Calculate the classification loss.
        loss = criterion(outputs, labels)

        # Backpropagation.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            dim=1
        )

        total_train += labels.size(0)

        correct_train += (
            predicted == labels
        ).sum().item()

    epoch_train_loss = (
        running_loss / len(train_loader)
    )

    epoch_train_accuracy = (
        100.0 * correct_train / total_train
    )

    train_losses.append(
        epoch_train_loss
    )

    train_accuracies.append(
        epoch_train_accuracy
    )


    # --------------------------------------------------------
    # Validation phase
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs.data,
                dim=1
            )

            total_val += labels.size(0)

            correct_val += (
                predicted == labels
            ).sum().item()

    epoch_val_loss = (
        val_loss / len(val_loader)
    )

    epoch_val_accuracy = (
        100.0 * correct_val / total_val
    )

    val_losses.append(
        epoch_val_loss
    )

    val_accuracies.append(
        epoch_val_accuracy
    )


    # Save the model only when validation accuracy improves.
    if epoch_val_accuracy > best_val_acc:
        best_val_acc = epoch_val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "New best model saved with "
            f"Validation Accuracy: {best_val_acc:.2f}%"
        )


    print(
        f"Epoch {epoch + 1} Summary | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Train Accuracy: {epoch_train_accuracy:.2f}% | "
        f"Validation Loss: {epoch_val_loss:.4f} | "
        f"Validation Accuracy: {epoch_val_accuracy:.2f}%"
    )


# ============================================================
# 9. Calculate total training time
# ============================================================

end_time = time.time()

elapsed_time = end_time - start_time

minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)


# ============================================================
# 10. Plot training results
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    train_losses,
    label="Train Loss",
    color="purple"
)

plt.plot(
    val_losses,
    label="Validation Loss",
    color="orange"
)

plt.title(
    "Loss Evolution - Baseline with Data Augmentation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    train_accuracies,
    label="Train Accuracy",
    color="purple"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy",
    color="orange"
)

plt.title(
    "Accuracy Evolution - Baseline with Data Augmentation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "training_results.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. Save experiment summary
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "training_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as summary_file:

    summary_file.write(
        "Experiment: Baseline CNN with Data Augmentation\n"
    )

    summary_file.write(
        f"Training Samples: {len(train_dataset)}\n"
    )

    summary_file.write(
        f"Validation Samples: {len(val_dataset)}\n"
    )

    summary_file.write(
        f"Batch Size: {batch_size}\n"
    )

    summary_file.write(
        f"Epochs: {epochs}\n"
    )

    summary_file.write(
        f"Learning Rate: {learning_rate}\n"
    )

    summary_file.write(
        f"Random Seed: {random_seed}\n"
    )

    summary_file.write(
        f"Training Time: {minutes}m {seconds}s\n"
    )

    summary_file.write(
        f"Best Validation Accuracy: {best_val_acc:.2f}%\n"
    )

    summary_file.write(
        f"Best Model Path: {BEST_MODEL_PATH}\n"
    )

    summary_file.write(
        "\nData Augmentation:\n"
    )

    summary_file.write(
        "- Random rotation: +/- 10 degrees\n"
    )

    summary_file.write(
        "- Random translation: up to 10%\n"
    )

    summary_file.write(
        "- Random scale: 90% to 110%\n"
    )

    summary_file.write(
        "- Random brightness change: 20%\n"
    )

    summary_file.write(
        "- Random contrast change: 20%\n"
    )


# ============================================================
# 12. Final output
# ============================================================

print("\nTraining finished!")

print(
    f"Results plot saved to: {plot_path}"
)

print(
    f"Best model saved to: {BEST_MODEL_PATH}"
)

print(
    f"Best Validation Accuracy: {best_val_acc:.2f}%"
)

print(
    f"Training Time: {minutes}m {seconds}s"
)