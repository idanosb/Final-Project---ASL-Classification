import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


if os.path.exists("/data/asl_alphabet_test"):
    test_dir = "/data/asl_alphabet_test"
else:
    test_dir = r"D:\idan\ASLData\asl_alphabet_test"

OUTPUT_DIR = "/results/VGG16_transfer_learning" if os.path.exists("/results") else "./VGG16_transfer_learning"
MODEL_PATH = os.path.join(OUTPUT_DIR, "VGG16_transfer_learning.pth")

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

num_classes = len(test_dataset.classes)

model = vgg16(weights=None)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)
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
plt.title("Confusion Matrix - VGG16 Transfer Learning")
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