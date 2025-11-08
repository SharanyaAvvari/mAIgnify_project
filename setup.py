"""
Quick Start Script - Automated Setup for mAIstro Medical Image Classifier
Run this to set up everything automatically
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description):
    """Run shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Step 1: Checking Python Version")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not Compatible")
        print("⚠️  Please install Python 3.8 or higher")
        return False

def install_dependencies():
    """Install required packages"""
    print_header("Step 2: Installing Dependencies")
    
    packages = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn[standard]", "ASGI server"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("pillow", "Image processing"),
        ("nibabel", "NIfTI file support"),
        ("scikit-learn", "Machine learning utilities"),
        ("matplotlib", "Plotting"),
    ]
    
    for package, description in packages:
        run_command(f"pip install {package}", f"Installing {description}")
    
    # PyTorch installation
    print("\n🔍 Detecting CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch already installed with CUDA")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
        else:
            print("✅ PyTorch already installed (CPU only)")
    except ImportError:
        print("📦 Installing PyTorch...")
        
        # Try to detect CUDA
        has_cuda = False
        try:
            result = subprocess.run("nvidia-smi", shell=True, 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                has_cuda = True
                print("✅ NVIDIA GPU detected - Installing PyTorch with CUDA support")
        except:
            print("ℹ️  No NVIDIA GPU detected - Installing CPU-only PyTorch")
        
        if has_cuda:
            run_command(
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "Installing PyTorch with CUDA"
            )
        else:
            run_command(
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
                "Installing PyTorch (CPU)"
            )

def create_directory_structure():
    """Create necessary directories"""
    print_header("Step 3: Creating Directory Structure")
    
    directories = [
        "uploads",
        "results",
        "medical_data/train/NORMAL",
        "medical_data/train/BENIGN",
        "medical_data/train/MALIGNANT",
        "medical_data/val/NORMAL",
        "medical_data/val/BENIGN",
        "medical_data/val/MALIGNANT",
        "medical_data/test/NORMAL",
        "medical_data/test/BENIGN",
        "medical_data/test/MALIGNANT",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_requirements_file():
    """Create requirements.txt"""
    print_header("Step 4: Creating requirements.txt")
    
    requirements = """# mAIstro Medical Image Classifier - Requirements

# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Data Processing
pandas==2.1.3
numpy==1.26.2

# Image Processing
pillow==10.1.0
nibabel==5.2.0
pydicom==2.4.3

# Machine Learning
torch==2.1.1
torchvision==0.16.1
scikit-learn==1.3.2

# Visualization
matplotlib==3.8.2

# Optional: For progress bars
tqdm==4.66.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_config_file():
    """Create configuration file"""
    print_header("Step 5: Creating Configuration File")
    
    config = """# mAIstro Configuration File
# Edit these settings as needed

[training]
batch_size = 32
num_epochs = 50
learning_rate = 0.001
train_split = 0.7
val_split = 0.15
test_split = 0.15

[model]
model_name = best_medical_model.pth
num_classes = 3
input_size = 224

[server]
host = 0.0.0.0
port = 8000

[paths]
upload_dir = uploads
results_dir = results
data_dir = medical_data
"""
    
    with open("config.ini", "w") as f:
        f.write(config)
    
    print("✅ Created config.ini")

def create_test_script():
    """Create test script"""
    print_header("Step 6: Creating Test Script")
    
    test_script = """#!/usr/bin/env python3
\"\"\"
Test Script for mAIstro Medical Image Classifier
Run this to test your setup
\"\"\"

import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def test_server_health():
    \"\"\"Test if server is running\"\"\"
    print("\\n🔍 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            print(f"   Status: {response.json()}")
            return True
        else:
            print("❌ Server returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print("   Make sure the server is running: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    \"\"\"Test model information endpoint\"\"\"
    print("\\n🔍 Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/api/model/info")
        if response.status_code == 200:
            info = response.json()
            print("✅ Model info retrieved")
            print(f"   Model loaded: {info['model_loaded']}")
            print(f"   Device: {info['device']}")
            print(f"   Method: {info['method']}")
            return True
        else:
            print("❌ Failed to get model info")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_file_upload():
    \"\"\"Test file upload with dummy image\"\"\"
    print("\\n🔍 Testing file upload...")
    
    # Create a dummy test image
    from PIL import Image
    import numpy as np
    import io
    
    # Create random image
    img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/upload",
            files={'file': ('test_image.png', img_bytes, 'image/png')}
        )
        
        if response.status_code == 200:
            file_info = response.json()
            print("✅ File upload successful")
            print(f"   File ID: {file_info['file_id']}")
            return file_info['file_id']
        else:
            print("❌ File upload failed")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_classification(file_id):
    \"\"\"Test image classification\"\"\"
    print("\\n🔍 Testing classification...")
    
    try:
        # Submit job
        response = requests.post(
            f"{BASE_URL}/api/submit",
            json={
                'prompt': 'Classify this medical image as malignant or benign',
                'user_id': 'test_user',
                'file_ids': [file_id]
            }
        )
        
        if response.status_code != 200:
            print("❌ Job submission failed")
            return False
        
        job_id = response.json()['job_id']
        print(f"✅ Job submitted: {job_id}")
        
        # Wait for completion
        print("   Waiting for processing...")
        max_wait = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
            job = response.json()
            
            if job['status'] == 'completed':
                print("✅ Classification completed")
                print(f"   Result: {job['result']['summary']}")
                if 'details' in job['result']:
                    details = job['result']['details']
                    if 'classification' in details:
                        print(f"   Classification: {details['classification']}")
                        print(f"   Confidence: {details['confidence']:.1%}")
                return True
            elif job['status'] == 'failed':
                print(f"❌ Classification failed: {job.get('error', 'Unknown error')}")
                return False
            
            time.sleep(1)
        
        print("❌ Classification timed out")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    \"\"\"Run all tests\"\"\"
    print("="*70)
    print("  mAIstro Medical Image Classifier - Test Suite")
    print("="*70)
    
    # Run tests
    tests_passed = 0
    tests_total = 4
    
    if test_server_health():
        tests_passed += 1
    
    if test_model_info():
        tests_passed += 1
    
    file_id = test_file_upload()
    if file_id:
        tests_passed += 1
        
        if test_classification(file_id):
            tests_passed += 1
    
    # Summary
    print("\\n" + "="*70)
    print(f"  TEST RESULTS: {tests_passed}/{tests_total} passed")
    print("="*70)
    
    if tests_passed == tests_total:
        print("\\n✅ All tests passed! Your setup is working correctly.")
        return 0
    else:
        print(f"\\n⚠️  {tests_total - tests_passed} test(s) failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    # Make executable on Unix systems
    try:
        os.chmod("test_setup.py", 0o755)
    except:
        pass
    
    print("✅ Created test_setup.py")

def create_readme():
    """Create README file"""
    print_header("Step 7: Creating README")
    
    readme = """# mAIstro Medical Image Classifier

AI-powered medical image classification system for cancer detection.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your medical images in the following structure:
```
medical_data/
├── train/
│   ├── NORMAL/
│   ├── BENIGN/
│   └── MALIGNANT/
├── val/
│   ├── NORMAL/
│   ├── BENIGN/
│   └── MALIGNANT/
└── test/
    ├── NORMAL/
    ├── BENIGN/
    └── MALIGNANT/
```

### 3. Train the Model
```bash
python train_model.py
```

This will create `best_medical_model.pth`

### 4. Start the Server
```bash
python main.py
```

### 5. Test the Setup
```bash
python test_setup.py
```

## API Endpoints

- `GET /` - API information
- `POST /api/upload` - Upload medical image
- `POST /api/submit` - Submit classification job
- `GET /api/jobs/{job_id}` - Get job status
- `GET /api/results/{job_id}/{filename}` - Download results
- `GET /api/health` - Health check
- `GET /api/model/info` - Model information

## Web Interface

Open your browser to: `http://localhost:8000` (if frontend is set up)

API Documentation: `http://localhost:8000/docs`

## Supported Formats

- NIfTI (.nii, .nii.gz)
- PNG (.png)
- JPEG (.jpg, .jpeg)
- DICOM (.dcm)

## Classes

1. **NORMAL** - Healthy tissue
2. **BENIGN** - Non-cancerous abnormalities
3. **MALIGNANT** - Cancerous lesions

## Important Notes

⚠️ **MEDICAL DISCLAIMER**: This system is for research and educational purposes only. 
It should NOT be used for actual medical diagnosis without proper validation by 
medical professionals and regulatory approval.

## Requirements

- Python 3.8+
- PyTorch
- FastAPI
- See `requirements.txt` for full list

## License

For educational and research purposes only.
"""
    
    with open("README.md", "w") as f:
        f.write(readme)
    
    print("✅ Created README.md")

def verify_setup():
    """Verify the setup is complete"""
    print_header("Step 8: Verifying Setup")
    
    checks = {
        "requirements.txt exists": os.path.exists("requirements.txt"),
        "config.ini exists": os.path.exists("config.ini"),
        "test_setup.py exists": os.path.exists("test_setup.py"),
        "README.md exists": os.path.exists("README.md"),
        "uploads/ directory exists": os.path.exists("uploads"),
        "results/ directory exists": os.path.exists("results"),
        "medical_data/ directory exists": os.path.exists("medical_data"),
    }
    
    all_passed = True
    for check, passed in checks.items():
        if passed:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
            all_passed = False
    
    return all_passed

def print_next_steps():
    """Print next steps for the user"""
    print_header("🎉 Setup Complete!")
    
    print("""
Your mAIstro Medical Image Classifier is now set up!

📋 NEXT STEPS:

1. 📊 Prepare Your Dataset
   - Collect medical images (at least 100 per class recommended)
   - Organize them in medical_data/ folders:
     • medical_data/train/NORMAL/
     • medical_data/train/BENIGN/
     • medical_data/train/MALIGNANT/
   - Similarly for val/ and test/ folders

2. 🎓 Train the Model
   python train_model.py
   
   This will:
   - Train a deep learning model on your data
   - Save the best model as 'best_medical_model.pth'
   - Take 30-120 minutes depending on your hardware

3. 🚀 Start the Server
   python main.py
   
   The server will start at http://localhost:8000

4. 🧪 Test the Setup
   python test_setup.py
   
   This will verify everything is working correctly

5. 📖 Read the Documentation
   - See README.md for detailed information
   - Check the setup guide for troubleshooting

📁 FILES CREATED:
   • requirements.txt - Python dependencies
   • config.ini - Configuration settings
   • test_setup.py - Automated testing script
   • README.md - Documentation
   • Directory structure for data and results

⚠️  IMPORTANT NOTES:
   • You MUST train the model before using it for classification
   • Without training, the system will use a fallback feature-based classifier
   • For best results, use at least 500 images per class
   • GPU is highly recommended for training (10-20x faster)

💡 HELPFUL COMMANDS:
   • Check GPU: nvidia-smi
   • View API docs: http://localhost:8000/docs (after starting server)
   • Check logs: tail -f server.log (if logging is enabled)

Need help? Check the troubleshooting section in the setup guide!
""")

def main():
    """Main setup function"""
    print("="*70)
    print("  mAIstro Medical Image Classifier - Quick Setup")
    print("="*70)
    print("\nThis script will set up everything you need to get started.\n")
    
    # Run setup steps
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return 1
    
    install_dependencies()
    create_directory_structure()
    create_requirements_file()
    create_config_file()
    create_test_script()
    create_readme()
    
    if verify_setup():
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Setup completed with warnings. Check the errors above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())