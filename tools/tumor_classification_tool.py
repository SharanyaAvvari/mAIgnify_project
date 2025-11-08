from smolagents import Tool
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

class TumorClassificationTool(Tool):
    """Classifies tumors as malignant or benign"""
    
    name = "tumor_classification"
    description = """Analyzes medical images to classify tumors.
    
    Inputs:
    - image_path: Path to tumor image (JPEG, PNG, DICOM)
    - model_path: Path to trained model
    
    Outputs:
    - classification: Malignant or Benign
    - confidence: Probability score
    - tumor_size: Estimated size in cm
    """
    
    inputs = {
        "image_path": {"type": "string"},
        "model_path": {"type": "string"}
    }
    
    output_type = "string"
    
    def forward(self, image_path: str, model_path: str) -> str:
        # Load model
        model = models.resnet50()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
        
        # Interpret results
        classes = ['Benign', 'Malignant']
        prediction = classes[pred.item()]
        confidence = probs[0][pred].item() * 100
        
        # Estimate tumor size (simplified - would need segmentation)
        tumor_size = "Requires segmentation for accurate measurement"
        
        result = f"""
Tumor Analysis Complete
{'='*50}

Classification: {prediction}
Confidence: {confidence:.1f}%

Note: This is an AI prediction. 
Please consult a medical professional for diagnosis.

{tumor_size}
"""
        return result.strip()