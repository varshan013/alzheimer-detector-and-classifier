import os
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    classes = os.listdir(source_dir)

    # Create output structure
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    # Loop for each class
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        print(f"{cls}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        # Copy images
        for f in train_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "train", cls))

        for f in val_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "val", cls))

        for f in test_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "test", cls))


if __name__ == "__main__":
    SOURCE = r"C:\Users\hp\OneDrive\Alzheimer's dataset\combined_images"         # your 4 folders inside this
    OUTPUT = r"alz_split_dataset"

    split_dataset(SOURCE, OUTPUT)
    print("Dataset split completed successfully!")
