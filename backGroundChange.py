import os
import cv2
import numpy as np
import random
from rembg import remove, new_session
from PIL import Image


class UniversalASLBlender:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.session = new_session("u2net", providers=["CPUExecutionProvider"])

    def _resize_to_target(self, img):
        return cv2.resize(img, self.target_size)

    def _ensure_transparency(self, pil_img):
        if pil_img.mode != "RGBA":
            return pil_img.convert("RGBA")
        return pil_img

    def remove_background(self, input_img_path):
        input_cv2 = cv2.imread(input_img_path)

        if input_cv2 is None:
            raise ValueError(f"Could not read image: {input_img_path}")

        input_cv2 = self._resize_to_target(input_cv2)

        input_pil = Image.fromarray(cv2.cvtColor(input_cv2, cv2.COLOR_BGR2RGB))
        input_pil = self._ensure_transparency(input_pil)

        output_pil = remove(input_pil, session=self.session)

        output_cv2 = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGBA2BGRA)
        return output_cv2

    def apply_random_background(self, hand_img_bgra, background_dirs):
        bg_files = []

        for bg_dir in background_dirs:
            if not os.path.exists(bg_dir):
                continue

            bg_files.extend([
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        if not bg_files:
            return None

        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path)

        if bg_img is None:
            raise ValueError(f"Could not read background: {bg_path}")

        bg_img = self._resize_to_target(bg_img)

        hand_img_bgra = hand_img_bgra.astype(np.float32)
        bg_img = bg_img.astype(np.float32)

        alpha = hand_img_bgra[:, :, 3] / 255.0
        foreground = hand_img_bgra[:, :, :3]

        mask_3ch = cv2.merge([alpha, alpha, alpha])

        foreground = cv2.multiply(foreground, mask_3ch)
        background_inv = cv2.multiply(bg_img, 1.0 - mask_3ch)

        final_img = cv2.add(foreground, background_inv)
        return final_img.astype(np.uint8)


# --- Configuration & Paths ---
RAW_ROOT_PATH = r"D:\idan\ASLData\asl_alphabet_train"
AUGMENTED_DATA_PATH = r"D:\idan\ASLDataWithBackGrounds"
BACKGROUNDS_DIRS = [r"D:\idan\BackGrounds"]

# מתחילים מ-C כי A ו-B כבר מוכנים
START_FROM = "C"

os.makedirs(AUGMENTED_DATA_PATH, exist_ok=True)

blender = UniversalASLBlender()

all_letters = sorted([
    folder for folder in os.listdir(RAW_ROOT_PATH)
    if os.path.isdir(os.path.join(RAW_ROOT_PATH, folder))
])

start_processing = False

print("Starting background removal and augmentation process (CPU Mode)...")

for current_letter in all_letters:
    if current_letter == START_FROM:
        start_processing = True

    if not start_processing:
        continue

    print(f"\n========== Processing class: {current_letter} ==========")

    raw_path = os.path.join(RAW_ROOT_PATH, current_letter)
    output_path = os.path.join(AUGMENTED_DATA_PATH, current_letter)

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(raw_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(raw_path, filename)
        output_filename = f"Augmented_{filename}"
        save_path = os.path.join(output_path, output_filename)

        if os.path.exists(save_path):
            print(f"Skipping existing: {save_path}")
            continue

        print(f"Processing: {current_letter}/{filename}")

        try:
            hand_transparent = blender.remove_background(input_path)
            final_augmented = blender.apply_random_background(
                hand_transparent,
                BACKGROUNDS_DIRS
            )

            if final_augmented is not None:
                cv2.imwrite(save_path, final_augmented)
                print(f"Saved: {save_path}")
            else:
                print(f"No backgrounds found in: {BACKGROUNDS_DIRS}")

        except Exception as e:
            print(f"Error processing {current_letter}/{filename}: {e}")

print("\nProcessing complete!")