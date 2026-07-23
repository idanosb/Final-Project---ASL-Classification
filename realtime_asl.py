from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = Path(
    r"D:\idan\FinalProject\results\VGG16_transfer_learning"
    r"\VGG16_transfer_learning.pth"
)

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 70.0
ROI_SIZE = 350

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]


# =========================================================
# Device selection
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================================================
# Validate model path
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found:\n{MODEL_PATH}"
    )


# =========================================================
# Image preprocessing
#
# This preprocessing must match the preprocessing used during
# training and evaluation.
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
# Build and load the VGG16 model
# =========================================================

model = models.vgg16(weights=None)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    len(CLASS_NAMES)
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded successfully.")


# =========================================================
# Open the webcam
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open the webcam. "
        "Try changing CAMERA_INDEX to 1 or 2."
    )


# =========================================================
# Real-time prediction loop
# =========================================================

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read a frame from the webcam.")
        break

    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # Calculate a fixed square region in the center of the frame
    center_x = frame_width // 2
    center_y = frame_height // 2

    half_roi = ROI_SIZE // 2

    x1 = max(0, center_x - half_roi)
    y1 = max(0, center_y - half_roi)
    x2 = min(frame_width, center_x + half_roi)
    y2 = min(frame_height, center_y + half_roi)

    # Crop the region of interest
    roi = frame[y1:y2, x1:x2]

    predicted_class = "No prediction"
    confidence = 0.0

    if roi.size > 0:
        # Convert OpenCV BGR image to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Convert NumPy image to PIL image
        pil_image = Image.fromarray(roi_rgb)

        # Apply the same preprocessing used during training
        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            confidence_tensor, predicted_index_tensor = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = predicted_index_tensor.item()
        confidence = confidence_tensor.item() * 100
        predicted_class = CLASS_NAMES[predicted_index]

    # Show the prediction only when confidence is high enough
    if confidence >= CONFIDENCE_THRESHOLD:
        display_text = f"{predicted_class} - {confidence:.2f}%"
    else:
        display_text = f"Uncertain - {confidence:.2f}%"

    # Draw the hand placement rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw a filled background behind the prediction text
    cv2.rectangle(
        frame,
        (20, 20),
        (420, 75),
        (0, 0, 0),
        -1
    )

    # Draw the prediction text
    cv2.putText(
        frame,
        display_text,
        (30, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Draw instructions
    cv2.putText(
        frame,
        "Place your hand inside the green square",
        (20, frame_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Display the webcam frame
    cv2.imshow("Real-Time ASL Recognition", frame)

    # Exit if the window was closed with the X button
    if cv2.getWindowProperty(
        "Real-Time ASL Recognition",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break

    key = cv2.waitKey(1) & 0xFF

    # Exit if the user presses Q
    if key == ord("q"):
        break


# =========================================================
# Cleanup
# =========================================================

camera.release()
cv2.destroyAllWindows()

print("Real-time ASL recognition stopped.")