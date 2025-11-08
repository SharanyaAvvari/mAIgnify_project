import os
import shutil
from pathlib import Path
import random

def organize_dataset(source_folder, output_folder, train_ratio=0.8):
    """
    Organize images into train/validation split
    
    Args:
        source_folder: Folder containing raw images
        output_folder: Output dataset folder
        train_ratio: Percentage for training (0.8 = 80%)
    """
    
    # Create output structure
    for split in ['train', 'validation']:
        for category in ['benign', 'malignant']:
            os.makedirs(f'{output_folder}/{split}/{category}', exist_ok=True)
    
    # Find all images
    benign_images = []
    malignant_images = []
    
    # Adjust these patterns based on your dataset
    for img_path in Path(source_folder).rglob('*.png'):
        img_str = str(img_path).lower()
        if 'benign' in img_str:
            benign_images.append(img_path)
        elif 'malignant' in img_str or 'tumor' in img_str:
            malignant_images.append(img_path)
    
    print(f"Found {len(benign_images)} benign images")
    print(f"Found {len(malignant_images)} malignant images")
    
    # Shuffle
    random.shuffle(benign_images)
    random.shuffle(malignant_images)
    
    # Split and copy
    def split_and_copy(images, category):
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"\nCopying {category} images...")
        for i, img in enumerate(train_images):
            shutil.copy(img, f'{output_folder}/train/{category}/')
            if i % 100 == 0:
                print(f"  Train: {i}/{len(train_images)}")
        
        for i, img in enumerate(val_images):
            shutil.copy(img, f'{output_folder}/validation/{category}/')
            if i % 100 == 0:
                print(f"  Val: {i}/{len(val_images)}")
        
        print(f"✅ {category.capitalize()}: {len(train_images)} train, {len(val_images)} val")
    
    split_and_copy(benign_images, 'benign')
    split_and_copy(malignant_images, 'malignant')
    
    print("\n✅ Dataset organized successfully!")
    print(f"\nDataset structure:")
    print(f"  train/benign: {len(os.listdir(f'{output_folder}/train/benign'))} images")
    print(f"  train/malignant: {len(os.listdir(f'{output_folder}/train/malignant'))} images")
    print(f"  validation/benign: {len(os.listdir(f'{output_folder}/validation/benign'))} images")
    print(f"  validation/malignant: {len(os.listdir(f'{output_folder}/validation/malignant'))} images")

# Usage
if __name__ == "__main__":
    # Change these paths to match your setup
    SOURCE = "dataset_raw/breakhis"  # Where you extracted dataset
    OUTPUT = "dataset"                # Output folder for training
    
    organize_dataset(SOURCE, OUTPUT, train_ratio=0.8)