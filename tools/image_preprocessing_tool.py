from smolagents import Tool
from PIL import Image
import numpy as np
import SimpleITK as sitk

class ImagePreprocessingTool(Tool):
    """Converts JPEG/PNG to format suitable for medical analysis"""
    
    name = "image_preprocessing"
    description = """Preprocesses images for medical analysis.
    
    Supports: JPEG, PNG, BMP, TIFF
    Converts to: NIfTI or NumPy array
    """
    
    inputs = {
        "image_path": {"type": "string"},
        "output_path": {"type": "string"}
    }
    
    output_type = "string"
    
    def forward(self, image_path: str, output_path: str) -> str:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L' and img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # If RGB, convert to grayscale for medical analysis
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # Normalize to 0-255
        img_array = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(img_array)
        
        # Set metadata (assuming 1mm spacing for 2D images)
        sitk_image.SetSpacing([1.0, 1.0])
        
        # Save as NIfTI
        output_nifti = output_path.replace('.jpg', '.nii.gz').replace('.png', '.nii.gz')
        sitk.WriteImage(sitk_image, output_nifti)
        
        return f"""
Image Preprocessing Complete
{'='*50}

Input: {image_path}
Format: {img.mode}
Size: {img.size}

Output: {output_nifti}
Format: NIfTI
Ready for: Medical image analysis, segmentation, classification

You can now use this image for:
- Tumor segmentation
- Classification
- Radiomic analysis
"""