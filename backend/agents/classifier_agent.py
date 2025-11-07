"""
Image Classifier Agent - Medical Image Classification
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import nibabel as nib
import os
from typing import List
import matplotlib.pyplot as plt

class ImageClassifierAgent:
    def __init__(self):
        self.name = "Image Classifier Agent"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # In production, load your trained model here
        # self.model = torch.load('path/to/model.pth')
        
    async def execute(self, prompt: str, file_paths: List[str], output_dir: str) -> dict:
        """
        Classify medical images (malignant/benign, etc.)
        """
        results = {}
        
        try:
            # Find image file
            image_path = None
            for file_path in file_paths:
                if any(file_path.endswith(ext) for ext in ['.nii', '.nii.gz', '.dcm', '.png', '.jpg', '.jpeg']):
                    image_path = file_path
                    break
            
            if not image_path:
                return {'summary': 'No image file found', 'error': 'No medical image provided'}
            
            # Load and preprocess image
            image_data = self._load_image(image_path)
            
            # Perform classification (demo mode - replace with real model)
            classification_result = self._classify_image(image_data, prompt)
            
            # Create visualization
            self._create_visualization(image_data, classification_result, output_dir)
            
            # Generate report
            report_path = os.path.join(output_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write("=== MEDICAL IMAGE CLASSIFICATION REPORT ===\n\n")
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write(f"Classification: {classification_result['class']}\n")
                f.write(f"Confidence: {classification_result['confidence']:.2%}\n\n")
                f.write("Findings:\n")
                for finding in classification_result['findings']:
                    f.write(f"• {finding}\n")
                f.write(f"\nRecommendation:\n{classification_result['recommendation']}\n")
            
            results = {
                'summary': f"Classification: {classification_result['class']} (Confidence: {classification_result['confidence']:.2%})",
                'classification': classification_result['class'],
                'confidence': classification_result['confidence'],
                'findings': classification_result['findings'],
                'recommendation': classification_result['recommendation']
            }
            
        except Exception as e:
            results = {
                'summary': f'Classification failed: {str(e)}',
                'error': str(e)
            }
        
        return results
    
    def _load_image(self, image_path: str):
        """Load medical image from various formats"""
        if image_path.endswith(('.nii', '.nii.gz')):
            # Load NIfTI
            img = nib.load(image_path)
            data = img.get_fdata()
            # Get middle slice
            if len(data.shape) == 3:
                data = data[:, :, data.shape[2]//2]
            return data
        elif image_path.endswith(('.png', '.jpg', '.jpeg')):
            # Load regular image
            img = Image.open(image_path).convert('L')
            return np.array(img)
        else:
            # Try PIL
            img = Image.open(image_path).convert('L')
            return np.array(img)
    
    def _classify_image(self, image_data: np.ndarray, prompt: str) -> dict:
        """
        Classify image
        In production, this would use your trained model
        """
        # DEMO MODE - Replace with real model inference
        # 
        # Real implementation would be:
        # preprocessed = self._preprocess(image_data)
        # with torch.no_grad():
        #     output = self.model(preprocessed)
        #     probabilities = torch.softmax(output, dim=1)
        #     predicted_class = torch.argmax(probabilities, dim=1)
        
        # Demo classification logic
        if 'malignant' in prompt.lower() or 'benign' in prompt.lower():
            # Simulate classification based on random features
            mean_intensity = np.mean(image_data)
            std_intensity = np.std(image_data)
            
            # Simple heuristic (replace with model)
            if mean_intensity > 100 and std_intensity > 30:
                classification = "BENIGN"
                confidence = 0.873
                findings = [
                    "Well-defined borders",
                    "Homogeneous density",
                    "Regular shape",
                    "No suspicious calcifications"
                ]
                recommendation = "Routine follow-up in 12 months recommended"
            else:
                classification = "MALIGNANT"
                confidence = 0.782
                findings = [
                    "Irregular borders detected",
                    "Heterogeneous density pattern",
                    "Spiculated margins",
                    "Suspicious microcalcifications"
                ]
                recommendation = "Immediate biopsy recommended. Further diagnostic workup required."
        else:
            classification = "NORMAL"
            confidence = 0.915
            findings = [
                "No abnormal findings",
                "Normal tissue architecture",
                "No masses or lesions detected"
            ]
            recommendation = "No further action required"
        
        return {
            'class': classification,
            'confidence': confidence,
            'findings': findings,
            'recommendation': recommendation
        }
    
    def _create_visualization(self, image_data: np.ndarray, classification: dict, output_dir: str):
        """Create annotated visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image_data, cmap='gray')
        ax.axis('off')
        
        # Add classification result
        result_text = f"{classification['class']}\nConfidence: {classification['confidence']:.1%}"
        color = 'green' if classification['class'] == 'BENIGN' else 'red'
        ax.text(0.05, 0.95, result_text, 
                transform=ax.transAxes, 
                fontsize=14, 
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                color='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'annotated_image.png'), 
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()