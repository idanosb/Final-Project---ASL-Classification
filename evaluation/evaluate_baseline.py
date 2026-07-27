import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if os.path.exists("/data/asl_alphabet_test"):
    test_dir = "/data/asl_alphabet_test"
else:
    test_dir = r"D:\idan\ASLData\asl_alphabet_test"

batch_size = 64
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "Enable GPU before running this script."
    )

device = torch.device("cuda")
# Must match the folder used during training
run_name = "baseline"
drive_output_dir = os.environ.get("ASL_OUTPUT_DIR")

if drive_output_dir:
    OUTPUT_DIR = os.path.join(drive_output_dir, run_name)
elif os.path.exists("/results"):
    OUTPUT_DIR = os.path.join("/results", run_name)
else:
    OUTPUT_DIR = PROJECT_ROOT / "results" / run_name


print("Results directory:", OUTPUT_DIR)

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_baseline_model.pth")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SimpleCNN(num_classes=len(test_dataset.classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print("========== TEST RESULTS ==========")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")
print("==================================")

summary_path = os.path.join(OUTPUT_DIR, "test_summary.txt")
with open(summary_path, "w") as f:
    f.write("========== TEST RESULTS ==========\n")
    f.write(f"Test Accuracy : {accuracy * 100:.2f}%\n")
    f.write(f"Precision     : {precision * 100:.2f}%\n")
    f.write(f"Recall        : {recall * 100:.2f}%\n")
    f.write(f"F1 Score      : {f1 * 100:.2f}%\n")

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
plt.imshow(cm)
plt.title("Confusion Matrix - Baseline CNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(test_dataset.classes)), test_dataset.classes, rotation=90)
plt.yticks(np.arange(len(test_dataset.classes)), test_dataset.classes)
plt.colorbar()
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)

print(f"Test summary saved to: {summary_path}")
print(f"Confusion matrix saved to: {cm_path}")
