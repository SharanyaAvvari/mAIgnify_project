"""
Data Organization Helper
Helps you organize and validate your medical image dataset
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

class DatasetOrganizer:
    """Helper to organize medical images into proper structure"""
    
    def __init__(self, output_dir='medical_data'):
        self.output_dir = output_dir
        self.supported_formats = ['.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.dcm']
        
    def create_structure(self):
        """Create the required directory structure"""
        splits = ['train', 'val', 'test']
        classes = ['NORMAL', 'BENIGN', 'MALIGNANT']
        
        for split in splits:
            for cls in classes:
                path = os.path.join(self.output_dir, split, cls)
                os.makedirs(path, exist_ok=True)
        
        print("✅ Directory structure created")
    
    def organize_from_single_folder(self, source_dir, class_name, 
                                     train_split=0.7, val_split=0.15):
        """
        Organize images from a single folder into train/val/test
        
        Args:
            source_dir: Folder containing images
            class_name: 'NORMAL', 'BENIGN', or 'MALIGNANT'
            train_split: Fraction for training (default 0.7)
            val_split: Fraction for validation (default 0.15)
        """
        if class_name not in ['NORMAL', 'BENIGN', 'MALIGNANT']:
            print(f"❌ Invalid class name: {class_name}")
            return
        
        # Get all image files
        files = []
        for ext in self.supported_formats:
            files.extend(Path(source_dir).glob(f'*{ext}'))
            files.extend(Path(source_dir).glob(f'**/*{ext}'))  # Recursive
        
        if not files:
            print(f"❌ No images found in {source_dir}")
            return
        
        # Shuffle
        random.shuffle(files)
        
        # Calculate splits
        n = len(files)
        train_n = int(n * train_split)
        val_n = int(n * val_split)
        
        train_files = files[:train_n]
        val_files = files[train_n:train_n+val_n]
        test_files = files[train_n+val_n:]
        
        # Copy files
        copied = {'train': 0, 'val': 0, 'test': 0}
        
        for split, file_list in [('train', train_files), 
                                  ('val', val_files), 
                                  ('test', test_files)]:
            dest_dir = os.path.join(self.output_dir, split, class_name)
            
            for file in file_list:
                dest_path = os.path.join(dest_dir, file.name)
                shutil.copy2(file, dest_path)
                copied[split] += 1
        
        print(f"\n✅ {class_name} organized:")
        print(f"   Train: {copied['train']} images")
        print(f"   Val: {copied['val']} images")
        print(f"   Test: {copied['test']} images")
        print(f"   Total: {sum(copied.values())} images")
    
    def organize_from_labeled_folders(self, source_dir, train_split=0.7, val_split=0.15):
        """
        Organize from pre-labeled folders
        
        Expected structure:
        source_dir/
            NORMAL/
                image1.nii
            BENIGN/
                image2.nii
            MALIGNANT/
                image3.nii
        """
        classes = ['NORMAL', 'BENIGN', 'MALIGNANT']
        
        for class_name in classes:
            class_dir = os.path.join(source_dir, class_name)
            if os.path.exists(class_dir):
                print(f"\n📂 Processing {class_name} folder...")
                self.organize_from_single_folder(class_dir, class_name, 
                                                train_split, val_split)
            else:
                print(f"⚠️  Folder not found: {class_dir}")
    
    def validate_dataset(self):
        """Validate the organized dataset"""
        print("\n" + "="*70)
        print("DATASET VALIDATION")
        print("="*70 + "\n")
        
        splits = ['train', 'val', 'test']
        classes = ['NORMAL', 'BENIGN', 'MALIGNANT']
        
        total_images = 0
        split_counts = defaultdict(int)
        class_counts = defaultdict(int)
        
        issues = []
        
        for split in splits:
            for class_name in classes:
                path = os.path.join(self.output_dir, split, class_name)
                
                if not os.path.exists(path):
                    issues.append(f"Missing directory: {path}")
                    continue
                
                # Count images
                count = 0
                for ext in self.supported_formats:
                    count += len(list(Path(path).glob(f'*{ext}')))
                
                split_counts[split] += count
                class_counts[class_name] += count
                total_images += count
                
                print(f"{split:6s} / {class_name:10s}: {count:4d} images")
        
        print("\n" + "-"*70)
        print("\nSUMMARY:")
        print(f"Total images: {total_images}")
        print(f"\nBy split:")
        for split in splits:
            print(f"  {split:6s}: {split_counts[split]:4d} images " +
                  f"({split_counts[split]/total_images*100:.1f}%)")
        
        print(f"\nBy class:")
        for class_name in classes:
            print(f"  {class_name:10s}: {class_counts[class_name]:4d} images " +
                  f"({class_counts[class_name]/total_images*100:.1f}%)")
        
        # Check for issues
        print("\n" + "-"*70)
        print("CHECKS:")
        
        # Minimum images per class
        min_per_class = 50
        for class_name in classes:
            if class_counts[class_name] < min_per_class:
                issues.append(f"{class_name} has only {class_counts[class_name]} images " +
                            f"(recommended: {min_per_class}+)")
        
        # Class balance
        if max(class_counts.values()) > 3 * min(class_counts.values()):
            issues.append("Classes are imbalanced (largest is 3x+ the smallest)")
        
        # Minimum total
        if total_images < 300:
            issues.append(f"Total images ({total_images}) is low (recommended: 300+)")
        
        if issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("\n✅ Dataset looks good!")
        
        print("\n" + "="*70)
        
        return len(issues) == 0
    
    def generate_statistics(self):
        """Generate detailed statistics about the dataset"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70 + "\n")
        
        splits = ['train', 'val', 'test']
        classes = ['NORMAL', 'BENIGN', 'MALIGNANT']
        
        # Image sizes and formats
        formats = defaultdict(int)
        sizes = []
        
        from PIL import Image
        import nibabel as nib
        
        for split in splits:
            for class_name in classes:
                path = os.path.join(self.output_dir, split, class_name)
                
                if not os.path.exists(path):
                    continue
                
                for file in Path(path).iterdir():
                    if file.is_file():
                        ext = ''.join(file.suffixes)
                        formats[ext] += 1
                        
                        try:
                            if ext in ['.nii', '.nii.gz']:
                                img = nib.load(str(file))
                                sizes.append(img.shape)
                            elif ext in ['.png', '.jpg', '.jpeg']:
                                img = Image.open(file)
                                sizes.append(img.size)
                        except:
                            pass
        
        print("FILE FORMATS:")
        for fmt, count in sorted(formats.items()):
            print(f"  {fmt:10s}: {count:4d} files")
        
        if sizes:
            print("\nIMAGE SIZES:")
            print(f"  Unique sizes: {len(set(sizes))}")
            if len(set(sizes)) <= 10:
                for size, count in defaultdict(int, 
                                              [(s, sizes.count(s)) for s in set(sizes)]).items():
                    print(f"    {str(size):20s}: {count:4d} images")
        
        print("\n" + "="*70)

def interactive_setup():
    """Interactive setup wizard"""
    print("="*70)
    print("  Medical Image Dataset Organizer")
    print("="*70)
    print("\nThis tool will help you organize your medical images.")
    print("\nOptions:")
    print("  1. I have images in separate folders (NORMAL, BENIGN, MALIGNANT)")
    print("  2. I have all images in one folder (need to label manually)")
    print("  3. Validate existing dataset")
    print("  4. Generate statistics")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    organizer = DatasetOrganizer()
    
    if choice == '1':
        source = input("\nEnter path to folder containing class folders: ").strip()
        if not os.path.exists(source):
            print(f"❌ Path not found: {source}")
            return
        
        organizer.create_structure()
        organizer.organize_from_labeled_folders(source)
        organizer.validate_dataset()
    
    elif choice == '2':
        print("\n⚠️  You'll need to manually label your images.")
        print("Place them in folders named: NORMAL, BENIGN, MALIGNANT")
        print("Then run this script again with option 1.")
    
    elif choice == '3':
        if organizer.validate_dataset():
            print("\n✅ Dataset is ready for training!")
        else:
            print("\n⚠️  Please address the issues above before training.")
    
    elif choice == '4':
        organizer.generate_statistics()
    
    else:
        print("❌ Invalid option")

# Example usage script
def example_usage():
    """Example of how to use the organizer"""
    print("""
# Example 1: Organize from labeled folders
organizer = DatasetOrganizer(output_dir='medical_data')
organizer.create_structure()
organizer.organize_from_labeled_folders('my_raw_images')
organizer.validate_dataset()

# Example 2: Organize single class
organizer = DatasetOrganizer()
organizer.create_structure()
organizer.organize_from_single_folder('normal_scans', 'NORMAL')
organizer.organize_from_single_folder('benign_scans', 'BENIGN')
organizer.organize_from_single_folder('malignant_scans', 'MALIGNANT')
organizer.validate_dataset()

# Example 3: Just validate
organizer = DatasetOrganizer()
organizer.validate_dataset()
organizer.generate_statistics()
    """)

if __name__ == "__main__":
    interactive_setup()