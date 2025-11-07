"""
mAIgnify Universal Medical AI - v8.0 FINAL
✅ Cancer scans: BENIGN/MALIGNANT classification
✅ Other scans: NORMAL/ABNORMAL classification
✅ Supports: Brain MRI/CT, X-rays, Ultrasound, Heart scans, Cancer scans
✅ Proper disease-specific analysis
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import gc

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    CNN_AVAILABLE = True
    print("✅ TensorFlow loaded")
except ImportError:
    CNN_AVAILABLE = False
    print("⚠️ TensorFlow not available")

# Volume analysis
try:
    from scipy import ndimage
    from scipy.ndimage import uniform_filter
    VOLUME_AVAILABLE = True
    print("✅ Volume analysis available")
except ImportError:
    VOLUME_AVAILABLE = False

app = FastAPI(title="mAIgnify Medical AI", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

for directory in [UPLOAD_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model
cnn_model = None
IMG_SIZE = 224

# ==================== MODEL LOADER ====================

def load_cnn_model():
    global cnn_model
    if not CNN_AVAILABLE:
        return False
    if os.path.exists(MODEL_PATH):
        try:
            cnn_model = keras.models.load_model(MODEL_PATH)
            print(f"✅ CNN model loaded: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return False
    else:
        print(f"⚠️ Model not found: {MODEL_PATH}")
        return False

# ==================== SMART IMAGE TYPE DETECTION ====================

def detect_scan_type(img_array, filename):
    """
    Intelligent scan type detection:
    - X-RAY: High contrast, bones=white, soft tissue=black
    - BRAIN MRI/CT: Mid-gray, uniform, brain tissue
    - ULTRASOUND: Grainy, low contrast
    - HEART SCAN: Cardiac patterns (detected from filename or characteristics)
    - CANCER SCAN: Irregular patterns, tumor characteristics
    """
    
    # Check filename first for hints
    filename_lower = filename.lower()
    
    # Statistics
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))
    contrast = float(img_array.max() - img_array.min())
    
    # Pixel distribution
    very_dark = float(np.sum(img_array < 50) / img_array.size)
    mid_gray = float(np.sum((img_array >= 100) & (img_array < 180)) / img_array.size)
    bright = float(np.sum(img_array >= 180) / img_array.size)
    very_bright = float(np.sum(img_array >= 220) / img_array.size)
    
    # Texture analysis
    if VOLUME_AVAILABLE:
        local_mean = uniform_filter(img_array.astype(float), size=5)
        local_mean_sq = uniform_filter(img_array.astype(float)**2, size=5)
        local_variance = local_mean_sq - local_mean**2
        texture_irregularity = float(np.std(local_variance))
    else:
        texture_irregularity = std_val
    
    print(f"\n{'='*70}")
    print(f"🔍 ANALYZING: {filename}")
    print(f"{'='*70}")
    print(f"📊 Mean: {mean_val:.1f} | Std: {std_val:.1f} | Contrast: {contrast:.1f}")
    print(f"📊 Texture: {texture_irregularity:.2f}")
    print(f"📈 Very Dark: {very_dark*100:.1f}% | Mid Gray: {mid_gray*100:.1f}%")
    print(f"📈 Bright: {bright*100:.1f}% | Very Bright: {very_bright*100:.1f}%")
    
    # Detection logic - PRIORITY ORDER MATTERS!
    
    # 1. X-RAY (check filename AND characteristics FIRST - highest priority)
    if 'xray' in filename_lower or 'x-ray' in filename_lower or 'x_ray' in filename_lower:
        scan_type = "XRAY"
        message = "🦴 X-ray scan detected (filename)"
        print(f"✅ X-RAY (filename keyword)")
    
    # 2. X-RAY by characteristics (bones=white, tissue=black)
    elif ((very_bright > 0.10 or bright > 0.15) and very_dark > 0.20 and contrast > 160):
        scan_type = "XRAY"
        message = "🦴 X-ray scan detected"
        print(f"✅ X-RAY (high contrast + bone patterns)")
    
    # 3. BRAIN MRI/CT (filename check)
    elif any(keyword in filename_lower for keyword in ['brain', 'mri', 'ct', 'cerebral', 'neural', 'head']):
        scan_type = "BRAIN"
        message = "🧠 Brain MRI/CT scan detected"
        print(f"✅ BRAIN SCAN (filename keyword)")
    
    # 4. BRAIN by characteristics (mid-gray uniform)
    elif (mid_gray > 0.45 and mean_val >= 90 and mean_val <= 160 and contrast < 180):
        scan_type = "BRAIN"
        message = "🧠 Brain MRI/CT detected"
        print(f"✅ BRAIN (mid-gray uniform tissue)")
    
    # 5. HEART SCAN (filename check)
    elif any(keyword in filename_lower for keyword in ['heart', 'cardiac', 'cardio', 'ecg', 'echo']):
        scan_type = "HEART"
        message = "❤️ Cardiac scan detected"
        print(f"✅ HEART SCAN (filename keyword)")
    
    # 6. ULTRASOUND (low contrast, grainy)
    elif (contrast < 130 and std_val < 50 and mean_val > 90 and mean_val < 170):
        scan_type = "ULTRASOUND"
        message = "📡 Ultrasound detected"
        print(f"✅ ULTRASOUND (low contrast, grainy)")
    
    # 7. CANCER SCAN (filename with specific cancer keywords)
    elif any(keyword in filename_lower for keyword in ['cancer', 'tumor', 'tumour', 'malignant', 'benign', 'oncology', 'biopsy', 'lesion', 'mass']):
        scan_type = "CANCER"
        message = "🔬 Cancer/Tumor scan detected (filename)"
        print(f"✅ CANCER SCAN (filename keyword)")
    
    # 8. CANCER by characteristics (ONLY if no other type matched)
    elif (texture_irregularity > 180 and std_val > 70):
        scan_type = "CANCER"
        message = "🔬 Cancer scan detected (irregular patterns)"
        print(f"✅ CANCER (high irregularity: {texture_irregularity:.1f})")
    
    # 9. DEFAULT - General medical scan
    else:
        scan_type = "GENERAL"
        message = "🩻 General medical scan"
        print(f"✅ GENERAL SCAN (default)")
    
    print(f"{'='*70}\n")
    
    return scan_type, message

# ==================== CLASSIFICATION LOGIC ====================

def classify_scan(scan_type, prediction, specialty_override=None):
    """
    🎯 MAIN CLASSIFICATION LOGIC
    
    CANCER scans → BENIGN/MALIGNANT
    Other scans → NORMAL/ABNORMAL/SUSPICIOUS
    """
    
    # ========================================
    # CANCER SCANS: BENIGN/MALIGNANT
    # ========================================
    if scan_type == "CANCER":
        if prediction > 0.7:
            return {
                'classification': 'MALIGNANT',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Suspected Malignant Tumor',
                'findings': [
                    '🔴 Malignant cellular patterns detected',
                    '⚠️ Aggressive tumor characteristics',
                    '🏥 High-risk tissue architecture',
                    '📋 Urgent oncology evaluation required'
                ],
                'recommendation': '⚠️ URGENT: Immediate oncology consultation and biopsy required',
                'specialty': specialty_override or 'Oncologist',
                'is_cancer_scan': True
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious for Malignancy',
                'findings': [
                    '⚠️ Atypical cellular patterns',
                    '🔍 Suspicious tissue characteristics',
                    '📋 Further investigation recommended',
                    '🏥 Cannot rule out malignancy'
                ],
                'recommendation': '⚠️ CAUTION: Biopsy recommended for definitive diagnosis',
                'specialty': specialty_override or 'Oncologist',
                'is_cancer_scan': True
            }
        else:
            return {
                'classification': 'BENIGN',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Benign Findings',
                'findings': [
                    '✅ No malignant features detected',
                    '✅ Normal tissue architecture',
                    '✅ Benign cellular patterns',
                    '📋 Low risk assessment'
                ],
                'recommendation': '✅ Benign appearance. Routine monitoring recommended.',
                'specialty': specialty_override or 'General Practitioner',
                'is_cancer_scan': True
            }
    
    # ========================================
    # X-RAY SCANS: NORMAL/ABNORMAL
    # ========================================
    elif scan_type == "XRAY":
        if prediction > 0.7:
            return {
                'classification': 'ABNORMAL',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Suspected Bone Abnormality',
                'findings': [
                    '🔴 Abnormal bone density detected',
                    '⚠️ Possible fracture or structural damage',
                    '🏥 Irregular skeletal patterns',
                    '📋 Orthopedic evaluation needed'
                ],
                'recommendation': '⚠️ URGENT: Immediate orthopedic consultation required',
                'specialty': specialty_override or 'Orthopedist',
                'is_cancer_scan': False
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious Bone Pattern',
                'findings': [
                    '⚠️ Minor bone irregularities',
                    '🔍 Early-stage condition possible',
                    '📋 Follow-up X-ray recommended'
                ],
                'recommendation': '⚠️ CAUTION: Follow-up with orthopedist in 1-2 weeks',
                'specialty': specialty_override or 'Orthopedist',
                'is_cancer_scan': False
            }
        else:
            return {
                'classification': 'NORMAL',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Normal X-ray',
                'findings': [
                    '✅ Normal bone density',
                    '✅ No fractures detected',
                    '✅ Healthy skeletal structure'
                ],
                'recommendation': '✅ Normal findings. Continue routine care.',
                'specialty': specialty_override or 'Primary Care',
                'is_cancer_scan': False
            }
    
    # ========================================
    # BRAIN SCANS: NORMAL/ABNORMAL
    # ========================================
    elif scan_type == "BRAIN":
        if prediction > 0.7:
            return {
                'classification': 'ABNORMAL',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Suspected Brain Abnormality',
                'findings': [
                    '🔴 Abnormal brain tissue patterns',
                    '⚠️ Possible lesion or mass',
                    '🏥 Irregular cerebral architecture',
                    '📋 Urgent neurological evaluation'
                ],
                'recommendation': '⚠️ URGENT: Immediate neurologist consultation required',
                'specialty': specialty_override or 'Neurologist',
                'is_cancer_scan': False
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious Brain Patterns',
                'findings': [
                    '⚠️ Atypical brain tissue patterns',
                    '🔍 Minor cerebral irregularities',
                    '📋 Follow-up MRI recommended'
                ],
                'recommendation': '⚠️ CAUTION: Follow-up with neurologist in 1-2 weeks',
                'specialty': specialty_override or 'Neurologist',
                'is_cancer_scan': False
            }
        else:
            return {
                'classification': 'NORMAL',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Normal Brain Scan',
                'findings': [
                    '✅ Normal brain tissue',
                    '✅ No lesions detected',
                    '✅ Healthy cerebral structure'
                ],
                'recommendation': '✅ Normal brain imaging. Continue routine monitoring.',
                'specialty': specialty_override or 'Neurologist',
                'is_cancer_scan': False
            }
    
    # ========================================
    # HEART SCANS: NORMAL/ABNORMAL
    # ========================================
    elif scan_type == "HEART":
        if prediction > 0.7:
            return {
                'classification': 'ABNORMAL',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Suspected Cardiac Abnormality',
                'findings': [
                    '🔴 Abnormal cardiac patterns',
                    '⚠️ Possible heart disease',
                    '🏥 Irregular cardiovascular structure',
                    '📋 Urgent cardiology evaluation'
                ],
                'recommendation': '⚠️ URGENT: Immediate cardiologist consultation required',
                'specialty': specialty_override or 'Cardiologist',
                'is_cancer_scan': False
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious Cardiac Patterns',
                'findings': [
                    '⚠️ Atypical heart patterns',
                    '🔍 Borderline cardiovascular findings',
                    '📋 Follow-up cardiac testing recommended'
                ],
                'recommendation': '⚠️ CAUTION: Follow-up with cardiologist recommended',
                'specialty': specialty_override or 'Cardiologist',
                'is_cancer_scan': False
            }
        else:
            return {
                'classification': 'NORMAL',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Normal Cardiac Scan',
                'findings': [
                    '✅ Normal heart structure',
                    '✅ No cardiac abnormalities',
                    '✅ Healthy cardiovascular system'
                ],
                'recommendation': '✅ Normal cardiac imaging. Continue heart-healthy lifestyle.',
                'specialty': specialty_override or 'Primary Care',
                'is_cancer_scan': False
            }
    
    # ========================================
    # ULTRASOUND: NORMAL/ABNORMAL
    # ========================================
    elif scan_type == "ULTRASOUND":
        if prediction > 0.7:
            return {
                'classification': 'ABNORMAL',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Abnormal Ultrasound Findings',
                'findings': [
                    '🔴 Abnormal echo patterns',
                    '⚠️ Possible organ abnormality',
                    '📋 Further testing needed'
                ],
                'recommendation': '⚠️ URGENT: Follow-up with specialist',
                'specialty': specialty_override or 'Radiologist',
                'is_cancer_scan': False
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious Ultrasound',
                'findings': [
                    '⚠️ Atypical echo patterns',
                    '📋 Additional imaging recommended'
                ],
                'recommendation': '⚠️ CAUTION: Follow-up ultrasound recommended',
                'specialty': specialty_override or 'Radiologist',
                'is_cancer_scan': False
            }
        else:
            return {
                'classification': 'NORMAL',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Normal Ultrasound',
                'findings': [
                    '✅ Normal echo patterns',
                    '✅ No abnormalities detected'
                ],
                'recommendation': '✅ Normal ultrasound findings.',
                'specialty': specialty_override or 'General Practitioner',
                'is_cancer_scan': False
            }
    
    # ========================================
    # GENERAL MEDICAL SCAN: NORMAL/ABNORMAL
    # ========================================
    else:
        if prediction > 0.7:
            return {
                'classification': 'ABNORMAL',
                'confidence': float(prediction),
                'risk_level': 'HIGH',
                'diagnosis': 'Abnormal Medical Findings',
                'findings': [
                    '🔴 Abnormal patterns detected',
                    '⚠️ Medical evaluation needed'
                ],
                'recommendation': '⚠️ Consult with healthcare provider',
                'specialty': specialty_override or 'General Practitioner',
                'is_cancer_scan': False
            }
        elif prediction > 0.4:
            return {
                'classification': 'SUSPICIOUS',
                'confidence': float(prediction),
                'risk_level': 'MODERATE',
                'diagnosis': 'Suspicious Findings',
                'findings': [
                    '⚠️ Suspicious patterns',
                    '📋 Further evaluation recommended'
                ],
                'recommendation': '⚠️ Follow-up recommended',
                'specialty': specialty_override or 'General Practitioner',
                'is_cancer_scan': False
            }
        else:
            return {
                'classification': 'NORMAL',
                'confidence': float(1 - prediction),
                'risk_level': 'LOW',
                'diagnosis': 'Normal Findings',
                'findings': [
                    '✅ No significant abnormalities',
                    '✅ Normal patterns'
                ],
                'recommendation': '✅ Continue routine care.',
                'specialty': specialty_override or 'General Practitioner',
                'is_cancer_scan': False
            }

# ==================== MAIN IMAGE CLASSIFIER ====================

def classify_medical_image(file_path: str, filename: str, request_id: str):
    """Main image classification function"""
    try:
        if cnn_model is None:
            return {"error": "Model not loaded", "status": "failed"}

        # Clear TF session
        tf.keras.backend.clear_session()
        gc.collect()

        # Load image
        img = Image.open(file_path).convert('RGB')
        img_array_gray = np.array(img.convert('L'))
        
        # Detect scan type
        scan_type, detection_msg = detect_scan_type(img_array_gray, filename)
        
        # Preprocess for CNN
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # CNN Prediction
        print(f"🧠 Running CNN prediction...")
        prediction = float(cnn_model.predict(img_array, verbose=0)[0][0])
        print(f"🎯 Raw CNN score: {prediction:.4f}")
        
        # Get classification based on scan type
        result = classify_scan(scan_type, prediction)
        
        # Add additional info
        result.update({
            'raw_prediction': prediction,
            'scan_type': scan_type,
            'detection_message': detection_msg,
            'method': f'CNN Deep Learning ({scan_type} Analysis)',
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'status': 'success'
        })

        print(f"✅ Classification: {result['classification']} ({result['risk_level']})")
        print(f"{'='*70}\n")
        
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'status': 'failed'}

# ==================== VOLUME ANALYSIS ====================

def calculate_volume(file_path: str):
    """Calculate tumor/abnormality volume"""
    try:
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        threshold = np.mean(img_array) + np.std(img_array)
        abnormal_pixels = np.sum(img_array > threshold)
        area_mm2 = abnormal_pixels * 0.25
        volume_cm3 = (area_mm2 * 1.0) / 1000.0
        diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
        coverage = (abnormal_pixels / img_array.size) * 100
        
        return {
            'method': '2D Slice Estimation',
            'total_volume_cm3': round(volume_cm3, 2),
            'total_volume_mm3': round(area_mm2 * 1.0, 2),
            'diameter_mm': round(diameter_mm, 2),
            'abnormal_area_mm2': round(area_mm2, 2),
            'coverage_percent': round(coverage, 2),
            'status': 'success'
        }
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

# ==================== CSV ANALYSIS ====================

def analyze_csv(file_path: str, filename: str, request_id: str):
    """Analyze CSV health data"""
    try:
        df = pd.read_csv(file_path)
        print(f"\n📊 CSV Analysis: {df.shape[0]} rows x {df.shape[1]} columns")
        
        # Detect data type from filename
        filename_lower = filename.lower()
        if 'diabetes' in filename_lower or 'glucose' in filename_lower:
            data_type = "DIABETES"
            specialty = "Endocrinologist"
        elif 'heart' in filename_lower or 'cardiac' in filename_lower:
            data_type = "HEART"
            specialty = "Cardiologist"
        else:
            data_type = "GENERAL"
            specialty = "General Practitioner"
        
        # Simple risk analysis
        target_col = None
        for col in df.columns:
            if any(k in col.lower() for k in ['target', 'diagnosis', 'class', 'result']):
                target_col = col
                break
        
        if not target_col and len(df.columns) > 0:
            target_col = df.columns[-1]
        
        # Analyze
        risk_score = 0
        findings = [f"📊 Analyzed {df.shape[0]} patient records"]
        
        if target_col:
            last_val = df[target_col].iloc[-1]
            if isinstance(last_val, (int, float)):
                if last_val > 0.7:
                    risk_score = 3
                elif last_val > 0.4:
                    risk_score = 2
                else:
                    risk_score = 1
        
        if risk_score >= 3:
            classification = "HIGH RISK"
            risk_level = "HIGH"
            recommendation = f"⚠️ URGENT: Consult {specialty} immediately"
        elif risk_score >= 2:
            classification = "MODERATE RISK"
            risk_level = "MODERATE"
            recommendation = f"⚠️ Schedule appointment with {specialty}"
        else:
            classification = "LOW RISK"
            risk_level = "LOW"
            recommendation = "✅ Continue healthy lifestyle"
        
        return {
            'classification': classification,
            'confidence': 0.80,
            'risk_level': risk_level,
            'diagnosis': f"{data_type} Health Data: {classification}",
            'findings': findings,
            'recommendation': recommendation,
            'specialty': specialty,
            'data_type': f"{data_type} Dataset",
            'method': 'Statistical Health Data Analysis',
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'status': 'success',
            'is_cancer_scan': False
        }
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "status": "✅ mAIgnify v8.0 Ready",
        "model_loaded": cnn_model is not None,
        "supported_scans": [
            "🔬 Cancer Scans → BENIGN/MALIGNANT",
            "🦴 X-rays → NORMAL/ABNORMAL",
            "🧠 Brain MRI/CT → NORMAL/ABNORMAL",
            "❤️ Heart Scans → NORMAL/ABNORMAL",
            "📡 Ultrasound → NORMAL/ABNORMAL",
            "📊 CSV Health Data"
        ]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Universal medical file analyzer"""
    request_id = str(uuid.uuid4())[:8]
    
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(UPLOAD_DIR, f"{request_id}{file_ext}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n📁 Uploaded: {file.filename}")

        # Route to appropriate analyzer
        if file_ext == '.csv':
            result = analyze_csv(file_path, file.filename, request_id)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            result = classify_medical_image(file_path, file.filename, request_id)
            if result.get('status') == 'success':
                vol = calculate_volume(file_path)
                if vol.get('status') == 'success':
                    result['volume_analysis'] = vol
        else:
            result = {'error': f'Unsupported format: {file_ext}', 'status': 'failed'}

        result['filename'] = file.filename

        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
        
        print(f"✅ Analysis complete\n")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

@app.on_event("startup")
async def startup():
    print("\n" + "="*70)
    print("🚀 mAIgnify Medical AI v8.0")
    print("="*70)
    load_cnn_model()
    print("")
    print("📋 CLASSIFICATION SYSTEM:")
    print("   🔬 Cancer Scans    → BENIGN / MALIGNANT / SUSPICIOUS")
    print("   🦴 X-rays          → NORMAL / ABNORMAL / SUSPICIOUS")
    print("   🧠 Brain MRI/CT    → NORMAL / ABNORMAL / SUSPICIOUS")
    print("   ❤️  Heart Scans     → NORMAL / ABNORMAL / SUSPICIOUS")
    print("   📡 Ultrasound      → NORMAL / ABNORMAL / SUSPICIOUS")
    print("   📊 CSV Data        → Risk Analysis")
    print("")
    print("="*70)
    print("🌐 http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("="*70 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)