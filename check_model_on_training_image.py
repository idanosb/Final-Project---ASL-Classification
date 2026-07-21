from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = Path(r"C:\Users\USER\ASLData\ASLData\asl_alphabet_train")

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "baseline"
    / "best_baseline_model.pth"
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "Enable GPU before running this script."
    )

DEVICE = torch.device("cuda")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

chosen_class = input("Enter a class, for example A or B: ").strip()

if chosen_class not in CLASSES:
    raise ValueError(f"Unknown class. Choose one of: {CLASSES}")

images = list((DATASET_DIR / chosen_class).glob("*.*"))

if not images:
    raise FileNotFoundError(f"No images found in {DATASET_DIR / chosen_class}")

image_path = random.choice(images)

model = SimpleCNN(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

image = Image.open(image_path).convert("RGB")

with torch.no_grad():
    output = model(transform(image).unsqueeze(0))
    probabilities = torch.softmax(output, dim=1)

confidence, predicted_index = torch.max(probabilities, dim=1)

print("\nImage file:", image_path.name)
print("True class:", chosen_class)
print("Prediction:", CLASSES[predicted_index.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")