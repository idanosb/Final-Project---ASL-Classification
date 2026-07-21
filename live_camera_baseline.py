from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# -------------------------------------------------
# 1. Same model architecture as base_line_model.py
# -------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

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
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------
# 2. Settings
# -------------------------------------------------
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. "
        "Enable GPU before running this script."
    )

device = torch.device("cuda")
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "results"
    / "baseline"
    / "best_baseline_model.pth"
)

HAND_MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"

# Screen behavior
DISPLAY_MIRROR = True

# If False:
# You see a mirrored screen, but the model gets the hand in the original direction.
# If predictions remain bad, change this to True and test again.
MODEL_SEES_MIRRORED_HAND = False

HAND_PADDING = 70


# -------------------------------------------------
# 3. Download MediaPipe hand model once
# -------------------------------------------------
if not HAND_MODEL_PATH.exists():
    print("Downloading MediaPipe hand detector model...")

    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )

    urllib.request.urlretrieve(model_url, HAND_MODEL_PATH)

    print("Hand detector model downloaded successfully.")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find trained model:\n{MODEL_PATH}\n\n"
        "Check that baseline_cpu_run_1 training finished successfully."
    )


# -------------------------------------------------
# 4. Same preprocessing as training
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# -------------------------------------------------
# 5. Load ASL model
# -------------------------------------------------
model = SimpleCNN(num_classes=len(CLASSES)).to(DEVICE)

try:
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
except TypeError:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)
model.eval()

print("Loaded model:", MODEL_PATH)
print("Using device:", DEVICE)
print("Press Q to close the camera.")
print("Press M to change whether the model sees a mirrored hand.")


# -------------------------------------------------
# 6. MediaPipe Hand Landmarker setup
# -------------------------------------------------
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)


# -------------------------------------------------
# 7. Open webcam
# -------------------------------------------------
camera = None

for camera_index in range(5):
    candidate = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if candidate.isOpened():
        camera = candidate
        print(f"Camera found at index: {camera_index}")
        break

    candidate.release()

if camera is None:
    raise RuntimeError(
        "Could not find an accessible camera. "
        "Close Camera, Zoom, Discord, Teams, or browser tabs using the webcam."
    )


# -------------------------------------------------
# 8. Live prediction
# -------------------------------------------------
start_time = time.time()

with vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    while True:
        success, original_frame = camera.read()

        if not success:
            print("Could not read frame from camera.")
            break

        # This is the frame shown to you.
        if DISPLAY_MIRROR:
            display_frame = cv2.flip(original_frame, 1)
        else:
            display_frame = original_frame.copy()

        frame_height, frame_width = display_frame.shape[:2]

        rgb_display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_display_frame,
        )

        timestamp_ms = int((time.time() - start_time) * 1000)

        hand_result = hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        predicted_label = "No hand"
        confidence = 0.0

        if hand_result.hand_landmarks:
            hand_landmarks = hand_result.hand_landmarks[0]

            x_values = [
                int(point.x * frame_width)
                for point in hand_landmarks
            ]

            y_values = [
                int(point.y * frame_height)
                for point in hand_landmarks
            ]

            x1 = max(0, min(x_values) - HAND_PADDING)
            y1 = max(0, min(y_values) - HAND_PADDING)
            x2 = min(frame_width, max(x_values) + HAND_PADDING)
            y2 = min(frame_height, max(y_values) + HAND_PADDING)

            # Make crop square
            crop_width = x2 - x1
            crop_height = y2 - y1
            crop_size = max(crop_width, crop_height)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(frame_width, x1 + crop_size)
            y2 = min(frame_height, y1 + crop_size)

            hand_crop = display_frame[y1:y2, x1:x2]

            if hand_crop.size > 0:
                # You can see a mirror, but the model can receive
                # either mirrored or original-looking crop.
                model_crop = hand_crop

                if DISPLAY_MIRROR and not MODEL_SEES_MIRRORED_HAND:
                    model_crop = cv2.flip(hand_crop, 1)

                rgb_hand_crop = cv2.cvtColor(
                    model_crop,
                    cv2.COLOR_BGR2RGB,
                )

                pil_hand_crop = Image.fromarray(rgb_hand_crop)

                image_tensor = transform(pil_hand_crop)
                image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)

                    confidence_tensor, prediction_tensor = torch.max(
                        probabilities,
                        dim=1,
                    )

                predicted_label = CLASSES[prediction_tensor.item()]
                confidence = confidence_tensor.item() * 100

                # Hand crop box
                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                # Draw the 21 hand landmarks
                for point in hand_landmarks:
                    point_x = int(point.x * frame_width)
                    point_y = int(point.y * frame_height)

                    cv2.circle(
                        display_frame,
                        (point_x, point_y),
                        3,
                        (255, 0, 0),
                        -1,
                    )

        mirror_status = "ON" if MODEL_SEES_MIRRORED_HAND else "OFF"

        cv2.putText(
            display_frame,
            f"Prediction: {predicted_label}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Confidence: {confidence:.1f}%",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Model mirror: {mirror_status} | M = toggle | Q = exit",
            (25, frame_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "ASL Live Recognition",
            display_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("m"):
            MODEL_SEES_MIRRORED_HAND = not MODEL_SEES_MIRRORED_HAND
            print(
                "MODEL_SEES_MIRRORED_HAND =",
                MODEL_SEES_MIRRORED_HAND,
            )


camera.release()
cv2.destroyAllWindows()