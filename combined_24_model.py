import os
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms


# ==================================================
# 1. Reproducibility
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# 2. Configuration
# ==================================================

COMMON_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(COMMON_CLASSES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

BATCH_SIZE = int(
    os.environ.get("ASL_BATCH_SIZE", "64")
)

EPOCHS = int(
    os.environ.get("ASL_EPOCHS", "3")
)

LEARNING_RATE = float(
    os.environ.get("ASL_LEARNING_RATE", "0.001")
)

NUM_WORKERS = int(
    os.environ.get("ASL_NUM_WORKERS", "2")
)

VALIDATION_SIZE = 0.20


# ==================================================
# 3. Read paths from environment variables
# ==================================================

required_environment_variables = [
    "ASL_TRAIN_DIR",
    "CABANA_TRAIN_DIR",
    "CABANA_TEST_DIR",
    "ASL_OUTPUT_DIR",
]

missing_variables = [
    variable
    for variable in required_environment_variables
    if not os.environ.get(variable)
]

if missing_variables:
    raise EnvironmentError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )

ASL_TRAIN_DIR = Path(
    os.environ["ASL_TRAIN_DIR"]
)

CABANA_TRAIN_DIR = Path(
    os.environ["CABANA_TRAIN_DIR"]
)

CABANA_TEST_DIR = Path(
    os.environ["CABANA_TEST_DIR"]
)

OUTPUT_DIR = (
    Path(os.environ["ASL_OUTPUT_DIR"])
    / "combined_24"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    OUTPUT_DIR
    / "best_combined_24_model.pth"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "combined_24_training_results.png"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "combined_24_summary.txt"
)

CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR
    / "cabana_test_classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "cabana_test_confusion_matrix.png"
)


# ==================================================
# 4. Select CPU or GPU automatically
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

pin_memory = device.type == "cuda"

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print(
        "Running on CPU. Training will work, "
        "but it may take considerably longer."
    )

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Workers:", NUM_WORKERS)
print("Output directory:", OUTPUT_DIR)


# ==================================================
# 5. Validate directories
# ==================================================

directories_to_check = {
    "Original ASL Train": ASL_TRAIN_DIR,
    "Cabana Train": CABANA_TRAIN_DIR,
    "Cabana Test": CABANA_TEST_DIR,
}

for name, directory in directories_to_check.items():
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{name} directory not found: {directory}"
        )

    print(f"{name}: {directory}")


# ==================================================
# 6. Collect image paths
# ==================================================

def collect_images(
    dataset_root: Path,
    source_name: str,
) -> list[dict]:

    samples = []

    for class_name in COMMON_CLASSES:
        class_directory = (
            dataset_root / class_name
        )

        if not class_directory.is_dir():
            raise FileNotFoundError(
                f"Missing class folder "
                f"{class_name} in {dataset_root}"
            )

        class_images = [
            image_path
            for image_path in class_directory.rglob("*")
            if image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        print(
            f"{source_name:15s} "
            f"{class_name:2s}: "
            f"{len(class_images)}"
        )

        for image_path in class_images:
            samples.append({
                "path": image_path,
                "label": CLASS_TO_INDEX[class_name],
                "class_name": class_name,
                "source": source_name,
            })

    return samples


print("\nCollecting original ASL images...")

asl_samples = collect_images(
    ASL_TRAIN_DIR,
    "original_asl",
)

print("\nCollecting Cabana Train images...")

cabana_train_samples = collect_images(
    CABANA_TRAIN_DIR,
    "cabana_train",
)

all_training_samples = (
    asl_samples
    + cabana_train_samples
)

print("\nOriginal ASL images:", len(asl_samples))
print(
    "Cabana Train images:",
    len(cabana_train_samples),
)
print(
    "Combined images:",
    len(all_training_samples),
)


# ==================================================
# 7. Stratified train-validation split
# ==================================================

all_indices = list(
    range(len(all_training_samples))
)

all_labels = [
    sample["label"]
    for sample in all_training_samples
]

train_indices, validation_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=all_labels,
)

train_samples = [
    all_training_samples[index]
    for index in train_indices
]

validation_samples = [
    all_training_samples[index]
    for index in validation_indices
]

print("\nTrain images:", len(train_samples))
print(
    "Validation images:",
    len(validation_samples),
)


# ==================================================
# 8. Transforms
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.RandomRotation(
        degrees=10,
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10),
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

evaluation_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ==================================================
# 9. Custom dataset
# ==================================================

class ASLCombinedDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        transform,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        sample = self.samples[index]

        image_path = sample["path"]
        label = sample["label"]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except Exception as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        return image, label


train_dataset = ASLCombinedDataset(
    train_samples,
    train_transform,
)

validation_dataset = ASLCombinedDataset(
    validation_samples,
    evaluation_transform,
)


# ==================================================
# 10. Weighted sampler
# ==================================================

train_class_counts = Counter(
    sample["label"]
    for sample in train_samples
)

source_counts = Counter(
    sample["source"]
    for sample in train_samples
)

print("\nTrain class counts:")

for class_index in range(
    len(COMMON_CLASSES)
):
    print(
        f"{INDEX_TO_CLASS[class_index]:2s}: "
        f"{train_class_counts[class_index]}"
    )

print("\nTrain source counts:")

for source_name, count in source_counts.items():
    print(source_name, ":", count)


sample_weights = []

for sample in train_samples:
    class_count = train_class_counts[
        sample["label"]
    ]

    source_count = source_counts[
        sample["source"]
    ]

    class_weight = 1.0 / class_count
    source_weight = 1.0 / source_count

    final_weight = (
        class_weight
        * source_weight
    )

    sample_weights.append(
        final_weight
    )


weighted_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),
    num_samples=len(train_samples),
    replacement=True,
)


# ==================================================
# 11. DataLoaders
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=weighted_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)


# ==================================================
# 12. CNN model with 24 outputs
# ==================================================

class CombinedCNN(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 8 * 8,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                256,
                number_of_classes,
            ),
        )

    def forward(
        self,
        inputs,
    ):
        features = self.features(inputs)
        return self.classifier(features)


model = CombinedCNN(
    number_of_classes=len(
        COMMON_CLASSES
    )
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
)


# ==================================================
# 13. Training
# ==================================================

train_losses = []
validation_losses = []
validation_accuracies = []

best_validation_accuracy = 0.0
training_start_time = time.time()

print(
    f"\nStarting combined training on {device}..."
)

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    print(
        f"\n--- Epoch {epoch + 1}/{EPOCHS} ---"
    )

    model.train()

    running_train_loss = 0.0
    train_image_count = 0

    for images, labels in train_loader:
        images = images.to(
            device,
            non_blocking=pin_memory,
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory,
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        current_batch_size = images.size(0)

        running_train_loss += (
            loss.item()
            * current_batch_size
        )

        train_image_count += (
            current_batch_size
        )

    epoch_train_loss = (
        running_train_loss
        / train_image_count
    )

    model.eval()

    running_validation_loss = 0.0
    validation_image_count = 0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(
                device,
                non_blocking=pin_memory,
            )

            labels = labels.to(
                device,
                non_blocking=pin_memory,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            predictions = outputs.argmax(
                dim=1
            )

            current_batch_size = (
                images.size(0)
            )

            running_validation_loss += (
                loss.item()
                * current_batch_size
            )

            validation_image_count += (
                current_batch_size
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

    epoch_validation_loss = (
        running_validation_loss
        / validation_image_count
    )

    epoch_validation_accuracy = (
        correct_predictions
        / validation_image_count
    )

    train_losses.append(
        epoch_train_loss
    )

    validation_losses.append(
        epoch_validation_loss
    )

    validation_accuracies.append(
        epoch_validation_accuracy
    )

    scheduler.step(
        epoch_validation_loss
    )

    if (
        epoch_validation_accuracy
        > best_validation_accuracy
    ):
        best_validation_accuracy = (
            epoch_validation_accuracy
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    COMMON_CLASSES,

                "epoch":
                    epoch + 1,

                "validation_accuracy":
                    best_validation_accuracy,
            },
            MODEL_PATH,
        )

        print(
            "New best model saved: "
            f"{best_validation_accuracy * 100:.2f}%"
        )

    epoch_time = (
        time.time()
        - epoch_start_time
    )

    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"Train Loss: "
        f"{epoch_train_loss:.4f} | "
        f"Val Loss: "
        f"{epoch_validation_loss:.4f} | "
        f"Val Acc: "
        f"{epoch_validation_accuracy * 100:.2f}% | "
        f"LR: "
        f"{current_learning_rate:.6f} | "
        f"Time: "
        f"{epoch_time / 60:.2f} min"
    )


training_time = (
    time.time()
    - training_start_time
)


# ==================================================
# 14. Plot training results
# ==================================================

epochs_range = range(
    1,
    EPOCHS + 1,
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs_range,
    train_losses,
    label="Train Loss",
)

plt.plot(
    epochs_range,
    validation_losses,
    label="Validation Loss",
)

plt.plot(
    epochs_range,
    validation_accuracies,
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Combined 24-Class Training")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 15. Load best model
# ==================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nBest model loaded from:",
    MODEL_PATH,
)


# ==================================================
# 16. Collect Cabana Test samples
# ==================================================

cabana_test_samples = collect_images(
    CABANA_TEST_DIR,
    "cabana_test",
)

cabana_test_dataset = ASLCombinedDataset(
    cabana_test_samples,
    evaluation_transform,
)

cabana_test_loader = DataLoader(
    cabana_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory,
)


# ==================================================
# 17. External evaluation on Cabana Test
# ==================================================

test_true_labels = []
test_predictions = []
test_confidences = []

with torch.no_grad():
    for images, labels in cabana_test_loader:
        images = images.to(
            device,
            non_blocking=pin_memory,
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidences, predictions = (
            probabilities.max(dim=1)
        )

        test_true_labels.extend(
            labels.tolist()
        )

        test_predictions.extend(
            predictions.cpu().tolist()
        )

        test_confidences.extend(
            confidences.cpu().tolist()
        )


test_accuracy = accuracy_score(
    test_true_labels,
    test_predictions,
)

test_precision = precision_score(
    test_true_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)

test_recall = recall_score(
    test_true_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)

test_f1 = f1_score(
    test_true_labels,
    test_predictions,
    average="weighted",
    zero_division=0,
)

average_confidence = float(
    np.mean(test_confidences)
)


print(
    "\n========== CABANA EXTERNAL TEST =========="
)

print(
    "Images:",
    len(test_true_labels),
)

print(
    f"Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{test_recall * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{test_f1 * 100:.2f}%"
)

print(
    f"Average confidence: "
    f"{average_confidence * 100:.2f}%"
)


# ==================================================
# 18. Classification report
# ==================================================

report = classification_report(
    test_true_labels,
    test_predictions,
    labels=list(
        range(len(COMMON_CLASSES))
    ),
    target_names=COMMON_CLASSES,
    zero_division=0,
)

with CLASSIFICATION_REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as report_file:
    report_file.write(report)

print("\nClassification report:")
print(report)


# ==================================================
# 19. Confusion matrix
# ==================================================

matrix = confusion_matrix(
    test_true_labels,
    test_predictions,
    labels=list(
        range(len(COMMON_CLASSES))
    ),
)

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    matrix,
    aspect="auto",
)

plt.title(
    "Combined 24-Class Model — Cabana Test"
)

plt.xlabel("Predicted class")
plt.ylabel("True class")

plt.xticks(
    np.arange(len(COMMON_CLASSES)),
    COMMON_CLASSES,
    rotation=90,
)

plt.yticks(
    np.arange(len(COMMON_CLASSES)),
    COMMON_CLASSES,
)

plt.colorbar()
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 20. Save summary
# ==================================================

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as summary_file:

    summary_file.write(
        "Combined 24-Class ASL Model\n"
    )

    summary_file.write(
        "===========================\n\n"
    )

    summary_file.write(
        f"Device: {device}\n"
    )

    summary_file.write(
        f"Epochs: {EPOCHS}\n"
    )

    summary_file.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    summary_file.write(
        f"Learning rate: "
        f"{LEARNING_RATE}\n"
    )

    summary_file.write(
        f"Original ASL samples: "
        f"{len(asl_samples)}\n"
    )

    summary_file.write(
        f"Cabana Train samples: "
        f"{len(cabana_train_samples)}\n"
    )

    summary_file.write(
        f"Training samples: "
        f"{len(train_samples)}\n"
    )

    summary_file.write(
        f"Validation samples: "
        f"{len(validation_samples)}\n"
    )

    summary_file.write(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%\n"
    )

    summary_file.write(
        f"Cabana Test accuracy: "
        f"{test_accuracy * 100:.2f}%\n"
    )

    summary_file.write(
        f"Cabana Test precision: "
        f"{test_precision * 100:.2f}%\n"
    )

    summary_file.write(
        f"Cabana Test recall: "
        f"{test_recall * 100:.2f}%\n"
    )

    summary_file.write(
        f"Cabana Test F1: "
        f"{test_f1 * 100:.2f}%\n"
    )

    summary_file.write(
        f"Average confidence: "
        f"{average_confidence * 100:.2f}%\n"
    )

    summary_file.write(
        f"Training time: "
        f"{training_time / 60:.2f} minutes\n"
    )


# ==================================================
# 21. Final paths
# ==================================================

print(
    "\n========== SAVED FILES =========="
)

print("Best model:", MODEL_PATH)
print("Training plot:", PLOT_PATH)
print("Summary:", SUMMARY_PATH)

print(
    "Classification report:",
    CLASSIFICATION_REPORT_PATH,
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH,
)

print(
    f"\nTotal training time: "
    f"{training_time / 60:.2f} minutes"
)