import os
import random
import shutil

TRAIN_DIR = r"D:\idan\ASLData\asl_alphabet_train"
TEST_DIR = r"D:\idan\ASLData\asl_alphabet_test"

TEST_RATIO = 0.10
SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(SEED)

os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(TRAIN_DIR):
    class_train_path = os.path.join(TRAIN_DIR, class_name)

    if not os.path.isdir(class_train_path):
        continue

    class_test_path = os.path.join(TEST_DIR, class_name)
    os.makedirs(class_test_path, exist_ok=True)

    images = [
        f for f in os.listdir(class_train_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    test_count = int(len(images) * TEST_RATIO)
    test_images = images[:test_count]

    print(f"{class_name}: moving {test_count} / {len(images)} images to test")

    for img_name in test_images:
        src = os.path.join(class_train_path, img_name)
        dst = os.path.join(class_test_path, img_name)

        if os.path.exists(dst):
            print(f"Skipping existing file: {dst}")
            continue

        shutil.move(src, dst)

print("\nDone! Train/Test split completed.")