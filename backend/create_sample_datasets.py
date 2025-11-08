"""
Create Sample Medical Datasets for Testing
Generates CSV files for Cancer, Heart Disease, and Diabetes
"""

import pandas as pd
import numpy as np
import os

# Create sample_data directory
os.makedirs('sample_data', exist_ok=True)

print("="*60)
print("Creating Sample Medical Datasets")
print("="*60)

# ==================== CANCER DATASET ====================

print("\n1️⃣  Creating Cancer Dataset...")

np.random.seed(42)
n_samples = 500

cancer_data = {
    'patient_id': [f'P{i:04d}' for i in range(n_samples)],
    'age': np.random.randint(30, 85, n_samples),
    'tumor_size_mm': np.random.uniform(5, 50, n_samples),
    'lymph_nodes_positive': np.random.randint(0, 15, n_samples),
    'metastasis': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'tumor_grade': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.4, 0.3]),
    'estrogen_receptor': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.7, 0.3]),
    'progesterone_receptor': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.6, 0.4]),
    'her2_status': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.2, 0.8]),
    'ki67_percentage': np.random.uniform(5, 80, n_samples),
    'tumor_stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
}

# Add diagnosis (target variable)
# Higher risk with: larger tumor, more lymph nodes, metastasis, higher grade
risk_score = (
    (cancer_data['tumor_size_mm'] / 50) * 0.3 +
    (cancer_data['lymph_nodes_positive'] / 15) * 0.3 +
    cancer_data['metastasis'] * 0.2 +
    (cancer_data['tumor_grade'] / 3) * 0.2
)

cancer_data['diagnosis'] = ['Malignant' if score > 0.5 else 'Benign' for score in risk_score]

cancer_df = pd.DataFrame(cancer_data)
cancer_df.to_csv('sample_data/cancer_dataset.csv', index=False)
print(f"✅ Created: sample_data/cancer_dataset.csv ({n_samples} samples)")
print(f"   Malignant: {(cancer_df['diagnosis'] == 'Malignant').sum()}")
print(f"   Benign: {(cancer_df['diagnosis'] == 'Benign').sum()}")

# ==================== HEART DISEASE DATASET ====================

print("\n2️⃣  Creating Heart Disease Dataset...")

np.random.seed(43)
n_samples = 500

heart_data = {
    'patient_id': [f'H{i:04d}' for i in range(n_samples)],
    'age': np.random.randint(25, 85, n_samples),
    'sex': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
    'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),
    'resting_blood_pressure': np.random.randint(90, 200, n_samples),
    'cholesterol_mg_dl': np.random.randint(100, 400, n_samples),
    'fasting_blood_sugar': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'resting_ecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1]),
    'max_heart_rate': np.random.randint(70, 202, n_samples),
    'exercise_angina': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'st_depression': np.random.uniform(0, 6, n_samples),
    'st_slope': np.random.choice([0, 1, 2], n_samples),
    'num_major_vessels': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'thalassemia': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.5, 0.3]),
}

# Calculate heart disease risk
risk_score = (
    (heart_data['age'] / 85) * 0.2 +
    (heart_data['cholesterol_mg_dl'] / 400) * 0.2 +
    (1 - heart_data['max_heart_rate'] / 202) * 0.2 +
    heart_data['exercise_angina'] * 0.15 +
    (heart_data['st_depression'] / 6) * 0.15 +
    (heart_data['num_major_vessels'] / 3) * 0.1
)

heart_data['heart_disease'] = ['High Risk' if score > 0.55 else 'Low Risk' for score in risk_score]

heart_df = pd.DataFrame(heart_data)
heart_df.to_csv('sample_data/heart_disease_dataset.csv', index=False)
print(f"✅ Created: sample_data/heart_disease_dataset.csv ({n_samples} samples)")
print(f"   High Risk: {(heart_df['heart_disease'] == 'High Risk').sum()}")
print(f"   Low Risk: {(heart_df['heart_disease'] == 'Low Risk').sum()}")

# ==================== DIABETES DATASET ====================

print("\n3️⃣  Creating Diabetes Dataset...")

np.random.seed(44)
n_samples = 500

diabetes_data = {
    'patient_id': [f'D{i:04d}' for i in range(n_samples)],
    'age': np.random.randint(20, 80, n_samples),
    'sex': np.random.choice(['M', 'F'], n_samples),
    'bmi': np.random.uniform(18, 45, n_samples),
    'fasting_glucose_mg_dl': np.random.randint(70, 250, n_samples),
    'hba1c_percentage': np.random.uniform(4, 12, n_samples),
    'systolic_bp': np.random.randint(90, 180, n_samples),
    'diastolic_bp': np.random.randint(60, 120, n_samples),
    'triglycerides': np.random.randint(50, 400, n_samples),
    'hdl_cholesterol': np.random.randint(20, 80, n_samples),
    'ldl_cholesterol': np.random.randint(50, 200, n_samples),
    'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'physical_activity_hours_week': np.random.randint(0, 15, n_samples),
    'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2]),
}

# Calculate diabetes risk
risk_score = (
    (diabetes_data['fasting_glucose_mg_dl'] / 250) * 0.3 +
    (diabetes_data['hba1c_percentage'] / 12) * 0.3 +
    (diabetes_data['bmi'] / 45) * 0.2 +
    diabetes_data['family_history'] * 0.1 +
    (1 - diabetes_data['physical_activity_hours_week'] / 15) * 0.1
)

diabetes_data['diabetes_status'] = [
    'Diabetic' if score > 0.6 else 
    'Pre-Diabetic' if score > 0.4 else 
    'Normal' 
    for score in risk_score
]

diabetes_df = pd.DataFrame(diabetes_data)
diabetes_df.to_csv('sample_data/diabetes_dataset.csv', index=False)
print(f"✅ Created: sample_data/diabetes_dataset.csv ({n_samples} samples)")
print(f"   Diabetic: {(diabetes_df['diabetes_status'] == 'Diabetic').sum()}")
print(f"   Pre-Diabetic: {(diabetes_df['diabetes_status'] == 'Pre-Diabetic').sum()}")
print(f"   Normal: {(diabetes_df['diabetes_status'] == 'Normal').sum()}")

# ==================== SUMMARY ====================

print("\n" + "="*60)
print("✅ ALL DATASETS CREATED SUCCESSFULLY!")
print("="*60)
print("\nFiles created in sample_data/ folder:")
print("  1. cancer_dataset.csv")
print("  2. heart_disease_dataset.csv")
print("  3. diabetes_dataset.csv")
print("\nUse these prompts to test:")
print("\n📊 For Cancer CSV:")
print("   'Analyze this cancer dataset and predict risk'")
print("   'Perform cancer classification on this patient data'")
print("\n❤️  For Heart Disease CSV:")
print("   'Predict cardiovascular disease risk from this data'")
print("   'Analyze heart disease risk factors in this dataset'")
print("\n🩸 For Diabetes CSV:")
print("   'Predict diabetes risk from this patient data'")
print("   'Analyze glucose levels and diabetes status'")
print("\n" + "="*60)
print("📁 Upload these CSV files to test your system!")
print("="*60 + "\n")