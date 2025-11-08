import os

def verify_dataset(dataset_path='backend/dataset'):
    """Verify dataset is organized correctly"""
    
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    required_folders = [
        'train/benign',
        'train/malignant',
        'validation/benign',
        'validation/malignant'
    ]
    
    total_train = 0
    total_val = 0
    
    for folder in required_folders:
        path = os.path.join(dataset_path, folder)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ {folder:30s} {count:5d} images")
            if 'train' in folder:
                total_train += count
            else:
                total_val += count
        else:
            print(f"❌ {folder:30s} MISSING!")
            return False
    
    print("="*60)
    print(f"Total Training Images:   {total_train}")
    print(f"Total Validation Images: {total_val}")
    print(f"Total Images:            {total_train + total_val}")
    print("="*60)
    
    if total_train < 100:
        print("⚠️  WARNING: Less than 100 training images!")
        print("   Recommended: 500+ images per class")
    elif total_train < 500:
        print("⚠️  CAUTION: Less than 500 training images")
        print("   Results may be suboptimal")
    else:
        print("✅ Good dataset size!")
    
    if total_val < 40:
        print("⚠️  WARNING: Less than 40 validation images!")
    
    # Check balance - FIXED THIS SECTION
    train_benign = len([f for f in os.listdir(f'{dataset_path}/train/benign') 
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
    train_malignant = len([f for f in os.listdir(f'{dataset_path}/train/malignant') 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    ratio = train_benign / train_malignant if train_malignant > 0 else 0
    
    print(f"\nClass Balance: {ratio:.2f}:1 (benign:malignant)")
    if 0.5 <= ratio <= 2.0:
        print("✅ Well balanced!")
    else:
        print("⚠️  WARNING: Imbalanced dataset!")
        print("   Try to balance benign and malignant images")
    
    return True

if __name__ == "__main__":
    verify_dataset()