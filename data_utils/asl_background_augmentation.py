import os
import cv2
import numpy as np
import random
from rembg import remove, new_session
from PIL import Image
from tqdm import tqdm


class UniversalASLBlender:
    def __init__(
        self,
        target_size=(224, 224),
        min_visible_ratio=0.10,
        max_visible_ratio=0.35,
        alpha_threshold=20
    ):
        self.target_size = target_size
        self.min_visible_ratio = min_visible_ratio
        self.max_visible_ratio = max_visible_ratio
        self.alpha_threshold = alpha_threshold

        # Force rembg to run on CPU to avoid GPU / driver issues.
        self.session = new_session("isnet-general-use", providers=["CPUExecutionProvider"])

        # Load background images once instead of scanning the folder for every image.
        self.background_files = []

    def load_backgrounds(self, background_dirs):
        """Load all available background file paths once."""
        bg_files = []

        for bg_dir in background_dirs:
            if not os.path.exists(bg_dir):
                print(f"Warning: background directory does not exist: {bg_dir}")
                continue

            bg_files.extend([
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        if not bg_files:
            raise ValueError(f"No background images found in: {background_dirs}")

        self.background_files = bg_files
        print(f"Loaded {len(self.background_files)} background images.")

    def _resize_to_target(self, img):
        """Resize image to the target dimensions."""
        return cv2.resize(img, self.target_size)

    def _ensure_transparency(self, pil_img):
        """Ensure PIL image has an alpha channel."""
        if pil_img.mode != "RGBA":
            return pil_img.convert("RGBA")
        return pil_img

    def _validate_alpha_mask(self, bgra_img, input_img_path):
   
        alpha = bgra_img[:, :, 3]

        visible_pixels = np.sum(alpha > self.alpha_threshold)
        total_pixels = alpha.shape[0] * alpha.shape[1]
        visible_ratio = visible_pixels / total_pixels

        ys, xs = np.where(alpha > self.alpha_threshold)

        if len(xs) == 0 or len(ys) == 0:
            raise ValueError(f"Bad mask: empty alpha mask, image={input_img_path}")

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_width = x_max - x_min + 1
        box_height = y_max - y_min + 1

        img_height, img_width = alpha.shape

        box_area_ratio = (box_width * box_height) / (img_width * img_height)

        if visible_ratio < self.min_visible_ratio:
            raise ValueError(
                f"Bad mask: hand almost disappeared. "
                f"visible_ratio={visible_ratio:.4f}, image={input_img_path}"
            )

        if visible_ratio > self.max_visible_ratio:
            raise ValueError(
                f"Bad mask: foreground is too large. "
                f"visible_ratio={visible_ratio:.4f}, image={input_img_path}"
            )

        if box_area_ratio > 0.50:
            raise ValueError(
                f"Bad mask: bounding box too large. "
                f"box_area_ratio={box_area_ratio:.4f}, image={input_img_path}"
            )

        touches_left = x_min <= 2
        touches_right = x_max >= img_width - 3
        touches_top = y_min <= 2
        touches_bottom = y_max >= img_height - 3

        edges_touched = sum([
            touches_left,
            touches_right,
            touches_top,
            touches_bottom
        ])

        if edges_touched >= 3:
            raise ValueError(
                f"Bad mask: mask touches too many image edges. "
                f"edges_touched={edges_touched}, image={input_img_path}"
            )

        return visible_ratio

    def remove_background(self, input_img_path):
        """Remove the original background and return a BGRA image."""
        input_cv2 = cv2.imread(input_img_path)

        if input_cv2 is None:
            raise ValueError(f"Could not read image: {input_img_path}")

        input_cv2 = self._resize_to_target(input_cv2)

        # Convert OpenCV BGR to PIL RGB.
        input_pil = Image.fromarray(cv2.cvtColor(input_cv2, cv2.COLOR_BGR2RGB))
        input_pil = self._ensure_transparency(input_pil)

        # Remove background using rembg.
        output_pil = remove(input_pil, session=self.session)

        # Convert PIL RGBA back to OpenCV BGRA.
        output_cv2 = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGBA2BGRA)

        # Validate that the hand was not removed.
        visible_ratio = self._validate_alpha_mask(output_cv2, input_img_path)

        return output_cv2, visible_ratio

    def apply_random_background(self, hand_img_bgra):
        """Blend the transparent hand onto a randomly selected background."""
        if not self.background_files:
            raise ValueError("No background images were loaded.")

        bg_path = random.choice(self.background_files)
        bg_img = cv2.imread(bg_path)

        if bg_img is None:
            raise ValueError(f"Could not read background: {bg_path}")

        bg_img = self._resize_to_target(bg_img)

        # Convert to float32 for safe blending math.
        hand_img_bgra = hand_img_bgra.astype(np.float32)
        bg_img = bg_img.astype(np.float32)

        alpha_raw = hand_img_bgra[:, :, 3].astype(np.float32)

        # Strengthen weak alpha values so the hand is less transparent
        alpha_raw = np.clip(alpha_raw * 1.8, 0, 255)

        alpha = alpha_raw / 255.0
        foreground = hand_img_bgra[:, :, :3]

        mask_3ch = cv2.merge([alpha, alpha, alpha])

        foreground = cv2.multiply(foreground, mask_3ch)
        background_inv = cv2.multiply(bg_img, 1.0 - mask_3ch)

        final_img = cv2.add(foreground, background_inv)
        return final_img.astype(np.uint8), bg_path


def get_classes_to_process(raw_root_path, start_from="C", skip_classes=None):
    """Return class folders starting from a selected class while skipping specific classes."""
    if skip_classes is None:
        skip_classes = set()

    all_classes = [
        folder for folder in os.listdir(raw_root_path)
        if os.path.isdir(os.path.join(raw_root_path, folder))
    ]

    # ASL alphabet order, including the common extra classes.
    preferred_order = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

    ordered_classes = [c for c in preferred_order if c in all_classes]
    remaining_classes = sorted([c for c in all_classes if c not in ordered_classes])
    ordered_classes.extend(remaining_classes)

    classes_to_process = []
    start_processing = False

    for class_name in ordered_classes:
        if class_name in skip_classes:
            continue

        if class_name == start_from:
            start_processing = True

        if start_processing:
            classes_to_process.append(class_name)

    if not classes_to_process:
        raise ValueError(
            f"No classes found to process. Check START_FROM='{start_from}' "
            f"and RAW_ROOT_PATH='{raw_root_path}'."
        )

    return classes_to_process


def write_failed_log(failed_log_path, class_name, filename, error_message):
    """Append failed image information to a log file."""
    with open(failed_log_path, "a", encoding="utf-8") as f:
        f.write(f"{class_name}/{filename} | {error_message}\n")


def main():
    # --- Configuration & Paths ---
    RAW_ROOT_PATH = os.environ.get("ASL_TRAIN_DIR") or (
        "/data/asl_alphabet_train"
        if os.path.exists("/data/asl_alphabet_train")
        else r"D:\idan\ASLData\asl_alphabet_train"
    )
    AUGMENTED_DATA_PATH = os.environ.get("ASL_AUGMENTED_DIR") or (
        "/augmented"
        if os.path.exists("/augmented")
        else r"D:\idan\ASLDataWithBackGrounds"
    )

    backgrounds_env = os.environ.get("ASL_BACKGROUNDS_DIRS")
    if backgrounds_env:
        BACKGROUNDS_DIRS = backgrounds_env.split(os.pathsep)
    elif os.path.exists("/backgrounds"):
        BACKGROUNDS_DIRS = ["/backgrounds"]
    else:
        BACKGROUNDS_DIRS = [r"D:\idan\BackGrounds"]

    # Start from C because A and B were already processed.
    START_FROM = "B"

    # Extra safety: never process A and B in this run.
    SKIP_CLASSES = 'A'

    # Keep this as 1 for now. Increase later only after checking quality.
    AUGMENTATIONS_PER_IMAGE = 1

    # Images with a smaller visible ratio are considered failed.
    MIN_VISIBLE_RATIO = 0.05

    # Images with a huge foreground are also suspicious.
    MAX_VISIBLE_RATIO = 0.85

    FAILED_LOG_PATH = os.path.join(AUGMENTED_DATA_PATH, "failed_images.txt")

    os.makedirs(AUGMENTED_DATA_PATH, exist_ok=True)

    # Clear old failed log at the beginning of a new run.
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Failed images log\n")
        f.write("=================\n")

    blender = UniversalASLBlender(
        target_size=(224, 224),
        min_visible_ratio=MIN_VISIBLE_RATIO,
        max_visible_ratio=MAX_VISIBLE_RATIO,
        alpha_threshold=20
    )

    blender.load_backgrounds(BACKGROUNDS_DIRS)

    classes_to_process = get_classes_to_process(
        RAW_ROOT_PATH,
        start_from=START_FROM,
        skip_classes=SKIP_CLASSES
    )

    print("Classes to process:", classes_to_process)
    print("Starting background removal and augmentation process (CPU Mode)...")

    total_images = 0
    successful_images = 0
    failed_images = 0
    skipped_images = 0

    for current_class in classes_to_process:
        raw_path = os.path.join(RAW_ROOT_PATH, current_class)
        output_path = os.path.join(AUGMENTED_DATA_PATH, current_class)

        os.makedirs(output_path, exist_ok=True)

        image_files = [
            f for f in os.listdir(raw_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"\n========== Processing class: {current_class} ==========")

        for filename in tqdm(image_files, desc=f"Class {current_class}", unit="img"):
            input_path = os.path.join(raw_path, filename)

            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                total_images += 1

                name, ext = os.path.splitext(filename)

                if AUGMENTATIONS_PER_IMAGE == 1:
                    output_filename = f"Augmented_{filename}"
                else:
                    output_filename = f"Augmented_{name}_bg{aug_idx + 1}{ext}"

                save_path = os.path.join(output_path, output_filename)

                if os.path.exists(save_path):
                    skipped_images += 1
                    continue

                try:
                    hand_transparent, visible_ratio = blender.remove_background(input_path)

                    final_augmented, bg_path = blender.apply_random_background(
                        hand_transparent
                    )

                    saved = cv2.imwrite(save_path, final_augmented)

                    if not saved:
                        raise ValueError(f"Could not save output image: {save_path}")

                    successful_images += 1

                except Exception as e:
                    failed_images += 1
                    write_failed_log(
                        FAILED_LOG_PATH,
                        current_class,
                        filename,
                        str(e)
                    )

    processed_images = successful_images + failed_images
    success_rate = (successful_images / processed_images * 100) if processed_images > 0 else 0

    print("\n==========================================")
    print("Processing Complete")
    print("==========================================")
    print(f"Total attempts:     {total_images}")
    print(f"Successful:         {successful_images}")
    print(f"Failed:             {failed_images}")
    print(f"Skipped existing:   {skipped_images}")
    print(f"Success rate:       {success_rate:.2f}%")
    print(f"Failed log:         {FAILED_LOG_PATH}")
    print("==========================================")


if __name__ == "__main__":
    main()
