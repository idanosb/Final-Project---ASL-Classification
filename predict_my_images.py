import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import vgg16


# =========================================================
# Paths
# =========================================================
TEST_FOLDER = 'A'

if os.path.exists("/mydata"):
    base_dir = Path("/mydata")
    TRAIN_DIR = Path("/data")
    MODEL_PATH = Path("/app/results/VGG16_transfer_learning/VGG16_transfer_learning.pth")
else:
    base_dir = Path(r"D:\idan\MyaslData")
    TRAIN_DIR = Path(r"D:\idan\ASLData\asl_alphabet_train")
    MODEL_PATH = Path(r"D:\idan\FinalProject\results\VGG16_transfer_learning\VGG16_transfer_learning.pth")

if TEST_FOLDER is None:
    MY_IMAGES_DIR = base_dir
else:
    MY_IMAGES_DIR = base_dir / TEST_FOLDER


# =========================================================
# Settings
# =========================================================



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# =========================================================
# Check paths
# =========================================================

if not MY_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Could not find your images directory:\n{MY_IMAGES_DIR}"
    )

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Could not find the training directory:\n{TRAIN_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find the model:\n{MODEL_PATH}"
    )


# =========================================================
# Image preprocessing
# Must be identical to the VGG16 evaluation preprocessing
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Get class names in the same order used during training
# =========================================================

training_dataset = datasets.ImageFolder(TRAIN_DIR)
class_names = training_dataset.classes
num_classes = len(class_names)

print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")


# =========================================================
# Build and load VGG16
# =========================================================

model = vgg16(weights=None)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()


# =========================================================
# Find images
# =========================================================

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_paths = sorted([
    path
    for path in MY_IMAGES_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in valid_extensions
])

if not image_paths:
    raise RuntimeError(
        f"No supported images were found in:\n{MY_IMAGES_DIR}"
    )


# =========================================================
# Predict every image
# =========================================================

correct_predictions = 0

print("\n========== MY IMAGE RESULTS ==========\n")

with torch.no_grad():

    for image_path in image_paths:
        expected_class = image_path.parent.name
        image = Image.open(image_path).convert("RGB")
        debug_image = image.resize((224, 224))
        debug_image.save("/app/debug_input.jpg")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

        predicted_index = top_indices[0][0].item()
        predicted_class = class_names[predicted_index]
        confidence = top_probabilities[0][0].item() * 100

        is_correct = predicted_class == expected_class

        if is_correct:
            correct_predictions += 1

        print(f"Image      : {image_path.name}")
        print(f"Prediction : {predicted_class}")
        print(f"Confidence : {confidence:.2f}%")
        print(f"Expected   : {expected_class}")
        print(f"Correct    : {'YES' if is_correct else 'NO'}")

        print("Top 3 predictions:")

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        ):
            class_name = class_names[index.item()]
            probability_percent = probability.item() * 100

            print(
                f"  {class_name}: "
                f"{probability_percent:.2f}%"
            )

        print("--------------------------------------")


# =========================================================
# Final accuracy
# =========================================================

accuracy = correct_predictions / len(image_paths)

print("\n========== FINAL RESULT ==========")
print(f"Images tested : {len(image_paths)}")
print(f"Correct       : {correct_predictions}")
print(f"Accuracy      : {accuracy * 100:.2f}%")
print("==================================")