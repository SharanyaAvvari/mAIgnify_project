"""
backend/tools/pytorch_vgg16_inference_tool.py

PyTorch VGG16 Inference Tool
"""

from smolagents import Tool
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import logging
from pathlib import Path
import json
import torch.nn as nn


class PyTorchVGG16InferenceTool(Tool):
    """Tool for VGG16 inference."""
    
    name = "pytorch_vgg16_inference"
    description = "Runs inference with trained VGG16 models."
    
    inputs = {
        "model_path": {"type": "string"},
        "test_dir": {"type": "string"},
        "output_file": {"type": "string"},
        "class_mapping_file": {"type": "string", "nullable": True}
    }
    
    output_type = "string"
    
    def _init_(self):
        super()._init_()
        self.logger = logging.getLogger(_name_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, model_path: str, test_dir: str, output_file: str, 
                class_mapping_file: str = None) -> str:
        """Run VGG16 inference."""
        
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
        
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Inference
        test_path = Path(test_dir)
        image_files = list(test_path.glob('.jpg')) + list(test_path.glob('.png'))
        
        results = []
        with torch.no_grad():
            for img_path in image_files:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = output.argmax(1).item()
                
                results.append({
                    'image': img_path.name,
                    'predicted_class': idx_to_class[predicted_class] if idx_to_class else predicted_class,
                    'confidence': probabilities[0][predicted_class].item()
                })
        
        # Save
        df = pd.DataFrame(results)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        return str(output_path)