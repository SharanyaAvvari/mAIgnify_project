"""
backend/tools/pytorch_resnet_inference_tool.py

PyTorch ResNet Inference Tool
Run inference with trained ResNet models
"""

from smolagents import Tool
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import logging
from pathlib import Path
import json
from typing import Dict
import torch.nn as nn


class PyTorchResNetInferenceTool(Tool):
    """
    Tool for running inference with trained ResNet models.
    """
    
    name = "pytorch_resnet_inference"
    description = """
    Runs inference on images using trained ResNet models.
    
    Features:
    - Load trained ResNet models
    - Batch or single image inference
    - Probability predictions
    - Export results to CSV
    
    Parameters:
    - model_path: Path to trained model (.pt)
    - test_dir: Directory with test images
    - output_file: Path to save predictions CSV
    - architecture: ResNet variant (18/34/50/101/152)
    - class_mapping_file: Path to class mapping JSON
    """
    
    inputs = {
        "model_path": {"type": "string"},
        "test_dir": {"type": "string"},
        "output_file": {"type": "string"},
        "architecture": {"type": "string"},
        "class_mapping_file": {"type": "string", "nullable": True}
    }
    
    output_type = "string"
    
    def _init_(self):
        super()._init_()
        self.logger = logging.getLogger(_name_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self,
                model_path: str,
                test_dir: str,
                output_file: str,
                architecture: str,
                class_mapping_file: str = None) -> str:
        """Run inference with ResNet model."""
        
        # Load class mapping
        if class_mapping_file:
            with open(class_mapping_file) as f:
                class_to_idx = json.load(f)
                idx_to_class = {v: k for k, v in class_to_idx.items()}
        else:
            idx_to_class = None
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = len(idx_to_class) if idx_to_class else 2
        
        model = self._get_resnet_model(architecture, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Get test images
        test_path = Path(test_dir)
        image_files = list(test_path.glob('.jpg')) + list(test_path.glob('.png'))
        
        # Run inference
        results = []
        with torch.no_grad():
            for img_path in image_files:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = output.argmax(1).item()
                confidence = probabilities[0][predicted_class].item()
                
                result = {
                    'image': img_path.name,
                    'predicted_class': idx_to_class[predicted_class] if idx_to_class else predicted_class,
                    'confidence': confidence
                }
                
                # Add probabilities for all classes
                if idx_to_class:
                    for idx, class_name in idx_to_class.items():
                        result[f'prob_{class_name}'] = probabilities[0][idx].item()
                
                results.append(result)
        
        # Save results
        df = pd.DataFrame(results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Inference complete. Results saved to {output_path}")
        
        return str(output_path)
    
    def _get_resnet_model(self, architecture: str, num_classes: int):
        """Load ResNet model architecture."""
        if architecture == "18":
            model = models.resnet18(pretrained=False)
        elif architecture == "34":
            model = models.resnet34(pretrained=False)
        elif architecture == "50":
            model = models.resnet50(pretrained=False)
        elif architecture == "101":
            model = models.resnet101(pretrained=False)
        elif architecture == "152":
            model = models.resnet152(pretrained=False)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model