from smolagents import Tool
import SimpleITK as sitk
import numpy as np

class TumorMeasurementTool(Tool):
    """Measures tumor size from segmentation mask"""
    
    name = "tumor_measurement"
    description = """Measures tumor dimensions from segmentation mask.
    
    Inputs:
    - image_path: Path to medical scan (NIfTI, DICOM)
    - mask_path: Path to tumor segmentation mask
    
    Outputs:
    - volume: Tumor volume in cm³
    - max_diameter: Largest diameter in cm
    - surface_area: Surface area in cm²
    """
    
    inputs = {
        "image_path": {"type": "string"},
        "mask_path": {"type": "string"}
    }
    
    output_type = "string"
    
    def forward(self, image_path: str, mask_path: str) -> str:
        # Load image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Get voxel spacing (mm)
        spacing = image.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
        
        # Convert mask to numpy
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Calculate volume
        num_tumor_voxels = np.sum(mask_array > 0)
        volume_mm3 = num_tumor_voxels * voxel_volume
        volume_cm3 = volume_mm3 / 1000
        
        # Calculate max diameter
        # Find bounding box
        tumor_coords = np.where(mask_array > 0)
        if len(tumor_coords[0]) > 0:
            z_range = (tumor_coords[0].max() - tumor_coords[0].min()) * spacing[2]
            y_range = (tumor_coords[1].max() - tumor_coords[1].min()) * spacing[1]
            x_range = (tumor_coords[2].max() - tumor_coords[2].min()) * spacing[0]
            max_diameter_mm = max(z_range, y_range, x_range)
            max_diameter_cm = max_diameter_mm / 10
        else:
            max_diameter_cm = 0
        
        # Calculate surface area (approximate)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        surface_area_mm2 = label_shape_filter.GetPerimeter(1)
        surface_area_cm2 = surface_area_mm2 / 100
        
        result = f"""
Tumor Measurements
{'='*50}

Volume: {volume_cm3:.2f} cm³
Maximum Diameter: {max_diameter_cm:.2f} cm
Surface Area: {surface_area_cm2:.2f} cm²

Number of Voxels: {num_tumor_voxels:,}
Voxel Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm

Clinical Significance:
- Tumors < 3 cm: Usually considered small
- Tumors 3-5 cm: Medium size
- Tumors > 5 cm: Large, may require aggressive treatment

⚠️ Note: These measurements are automated. 
Medical professionals should verify for clinical decisions.
"""
        return result.strip()