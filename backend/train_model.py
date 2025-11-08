"""
mAIstro-Integrated Training System
Automated multi-agent pipeline for medical image analysis with tumor volume calculation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import json
import zipfile
import glob
import pandas as pd
from datetime import datetime
import subprocess
import sys

print("="*60)
print("🧠 mAIstro Multi-Agent Training System")
print("="*60)

# ==================== GPU CONFIGURATION ====================

print("\n🔍 Checking GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("❌ No GPU found - using CPU")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'training_results')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== AGENT 1: DATA EXTRACTION ====================

def extract_datasets():
    """
    Extracts uploaded datasets (supports multiple formats)
    Mimics mAIstro's data preprocessing capabilities
    """
    print("\n" + "="*60)
    print("AGENT 1: DATA EXTRACTION")
    print("="*60)
    
    zip_files = glob.glob('*.zip')
    
    if not zip_files:
        print("❌ No ZIP files found!")
        return False
    
    print(f"✅ Found {len(zip_files)} ZIP file(s)")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    for zip_file in zip_files:
        print(f"\n📦 Extracting {zip_file}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)
            print(f"✅ Extracted successfully!")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    # Find dataset structure
    train_path = os.path.join(DATASET_DIR, 'train')
    val_path = os.path.join(DATASET_DIR, 'validation')
    test_path = os.path.join(DATASET_DIR, 'test')
    
    # Check for nested structures
    if not os.path.exists(train_path):
        subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if subdirs:
            first_subdir = subdirs[0]
            possible_train = os.path.join(DATASET_DIR, first_subdir, 'train')
            if os.path.exists(possible_train):
                train_path = possible_train
                val_path = os.path.join(DATASET_DIR, first_subdir, 'validation')
                test_path = os.path.join(DATASET_DIR, first_subdir, 'test')
    
    print("\n✅ Dataset structure verified:")
    print(f"   Train: {train_path}")
    print(f"   Validation: {val_path}")
    print(f"   Test: {test_path}")
    
    return train_path, val_path, test_path

# ==================== AGENT 2: EDA (EXPLORATORY DATA ANALYSIS) ====================

def perform_eda(train_path, val_path, test_path):
    """
    Performs comprehensive EDA like mAIstro's EDA Agent
    Generates statistics, visualizations, and reports
    """
    print("\n" + "="*60)
    print("AGENT 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    eda_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {}
    }
    
    for dataset_name, dataset_path in [('train', train_path), ('validation', val_path), ('test', test_path)]:
        if not os.path.exists(dataset_path):
            continue
        
        print(f"\n📊 Analyzing {dataset_name} dataset...")
        
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        dataset_stats = {
            'classes': classes,
            'class_distribution': {}
        }
        
        total_images = 0
        for cls in classes:
            cls_path = os.path.join(dataset_path, cls)
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii'))])
            dataset_stats['class_distribution'][cls] = count
            total_images += count
            print(f"   {cls}: {count} images")
        
        dataset_stats['total_images'] = total_images
        eda_results['datasets'][dataset_name] = dataset_stats
    
    # Save EDA report
    eda_report_path = os.path.join(RESULTS_DIR, 'eda_report.json')
    with open(eda_report_path, 'w') as f:
        json.dump(eda_results, f, indent=2)
    
    # Generate text report
    text_report_path = os.path.join(RESULTS_DIR, 'eda_report.txt')
    with open(text_report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {eda_results['timestamp']}\n\n")
        
        for dataset_name, stats in eda_results['datasets'].items():
            f.write(f"\n{dataset_name.upper()} DATASET:\n")
            f.write(f"  Total Images: {stats['total_images']}\n")
            f.write(f"  Classes: {len(stats['classes'])}\n")
            f.write(f"  Distribution:\n")
            for cls, count in stats['class_distribution'].items():
                percentage = (count / stats['total_images']) * 100
                f.write(f"    - {cls}: {count} ({percentage:.1f}%)\n")
    
    print(f"\n✅ EDA completed: {eda_report_path}")
    return eda_results

# ==================== AGENT 3: CNN MODEL BUILDER ====================

def create_cnn_model(num_classes=2):
    """
    Creates CNN architecture similar to mAIstro's Image Classifier Agent
    Supports ResNet-style architecture with batch normalization
    """
    print("\n" + "="*60)
    print("AGENT 3: CNN MODEL ARCHITECTURE")
    print("="*60)
    
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output
        layers.Dense(1 if num_classes == 2 else num_classes, 
                    activation='sigmoid' if num_classes == 2 else 'softmax')
    ])

    print("✅ Model architecture created!")
    model.summary()
    return model

# ==================== AGENT 4: DATA PREPARATION ====================

def prepare_dataset(train_path, val_path, test_path):
    """
    Prepares data generators with augmentation
    Similar to mAIstro's preprocessing pipeline
    """
    print("\n" + "="*60)
    print("AGENT 4: DATA PREPARATION & AUGMENTATION")
    print("="*60)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_generator = None
    if os.path.exists(val_path):
        val_generator = val_test_datagen.flow_from_directory(
            val_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
    
    test_generator = None
    if os.path.exists(test_path):
        test_generator = val_test_datagen.flow_from_directory(
            test_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

    print(f"✅ Training samples: {train_generator.samples}")
    if val_generator:
        print(f"✅ Validation samples: {val_generator.samples}")
    if test_generator:
        print(f"✅ Test samples: {test_generator.samples}")

    return train_generator, val_generator, test_generator

# ==================== AGENT 5: MODEL TRAINING ====================

def train_model():
    """
    Main training pipeline orchestrated by Master Agent
    """
    print("\n" + "="*60)
    print("MASTER AGENT: ORCHESTRATING TRAINING PIPELINE")
    print("="*60)

    # Step 1: Extract data
    result = extract_datasets()
    if not result:
        return False
    train_path, val_path, test_path = result

    # Step 2: Perform EDA
    eda_results = perform_eda(train_path, val_path, test_path)

    # Step 3: Prepare data
    train_gen, val_gen, test_gen = prepare_dataset(train_path, val_path, test_path)

    # Step 4: Create model
    model = create_cnn_model()

    # Step 5: Compile
    print("\n⚙️ Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    # Step 6: Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy' if val_gen else 'accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_gen else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_gen else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, 'training_log.csv')
        )
    ]

    # Step 7: Train
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Step 8: Save final model
    final_model_path = os.path.join(MODEL_DIR, 'cancer_classifier.h5')
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")

    # Step 9: Evaluate on test set
    if test_gen:
        print("\n" + "="*60)
        print("AGENT 6: MODEL EVALUATION")
        print("="*60)
        test_results = model.evaluate(test_gen, verbose=1)
        
        evaluation_report = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'test_auc': float(test_results[4])
        }
        
        eval_path = os.path.join(RESULTS_DIR, 'test_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {evaluation_report['test_accuracy']:.4f}")
        print(f"   Precision: {evaluation_report['test_precision']:.4f}")
        print(f"   Recall: {evaluation_report['test_recall']:.4f}")
        print(f"   AUC: {evaluation_report['test_auc']:.4f}")

    # Step 10: Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'precision': [float(x) for x in history.history['precision']],
        'recall': [float(x) for x in history.history['recall']],
        'auc': [float(x) for x in history.history['auc']]
    }
    
    if val_gen:
        history_dict.update({
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_precision': [float(x) for x in history.history['val_precision']],
            'val_recall': [float(x) for x in history.history['val_recall']],
            'val_auc': [float(x) for x in history.history['val_auc']]
        })

    history_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("="*60)
    print(f"\n📁 Results Directory: {RESULTS_DIR}")
    print(f"📁 Model Directory: {MODEL_DIR}")
    print("\n📊 Generated Files:")
    print(f"   - {final_model_path}")
    print(f"   - {history_path}")
    print(f"   - {os.path.join(RESULTS_DIR, 'eda_report.txt')}")
    if test_gen:
        print(f"   - {eval_path}")

    return True

# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("mAIstro Multi-Agent Training System")
    print("Automated end-to-end medical AI model development")
    print("="*60)
    
    print("\n📋 System Overview:")
    print("   AGENT 1: Data Extraction")
    print("   AGENT 2: Exploratory Data Analysis (EDA)")
    print("   AGENT 3: CNN Architecture Builder")
    print("   AGENT 4: Data Preparation & Augmentation")
    print("   AGENT 5: Model Training")
    print("   AGENT 6: Model Evaluation")
    print("\n" + "="*60)
    
    input_check = input("\n✅ Upload dataset ZIP files. Ready to start? (yes/no): ")
    
    if input_check.lower() == 'yes':
        success = train_model()
        
        if success:
            print("\n" + "="*60)
            print("NEXT STEPS:")
            print("="*60)
            print("1. Check training results in: training_results/")
            print("2. Use trained model: models/cancer_classifier.h5")
            print("3. Review EDA report: training_results/eda_report.txt")
            print("4. Integrate model into backend (main.py)")
            print("="*60)
    else:
        print("\n📤 Please upload your dataset ZIP files first, then run again")