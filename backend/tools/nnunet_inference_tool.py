"""
nnU-Net Inference Tool
Performs inference using trained nnU-Net models
"""

from smolagents import Tool
import subprocess
import os
from pathlib import Path
from typing import Optional


class NNUNetInferenceTool(Tool):
    """
    Tool for nnU-Net model inference
    """
    
    name = "nnunet_inference"
    description = """Performs inference using trained nnU-Net models.
    
    Inputs:
    - input_folder: Folder containing images to segment
    - output_folder: Folder to save segmentation masks
    - dataset_id: Dataset ID used during training
    - configuration: Model configuration ('2d', '3d_fullres', etc.)
    - fold: Fold used during training ('all', '0', '1', etc.)
    - checkpoint: Checkpoint to use ('checkpoint_best', 'checkpoint_final')
    - use_tta: Use test-time augmentation (default: False)
    
    Outputs:
    - Segmentation masks (NIfTI format)
    - Inference summary
    """
    
    inputs = {
        "input_folder": {"type": "string", "description": "Input images folder"},
        "output_folder": {"type": "string", "description": "Output folder"},
        "dataset_id": {"type": "string", "description": "Dataset ID"},
        "configuration": {"type": "string", "description": "Config", "default": "3d_fullres"},
        "fold": {"type": "string", "description": "Fold", "default": "all"},
        "checkpoint": {"type": "string", "description": "Checkpoint name", "default": "checkpoint_best"},
        "use_tta": {"type": "boolean", "description": "Test-time augmentation", "default": False}
    }
    
    output_type = "string"
    
    def forward(self, input_folder: str, output_folder: str, dataset_id: str,
                configuration: str = "3d_fullres", fold: str = "all",
                checkpoint: str = "checkpoint_best", use_tta: bool = False) -> str:
        """
        Run nnU-Net inference
        
        Args:
            input_folder: Path to input images
            output_folder: Path to save outputs
            dataset_id: Dataset identifier
            configuration: Model configuration
            fold: Training fold
            checkpoint: Checkpoint file name
            use_tta: Use test-time augmentation
            
        Returns:
            Inference summary
        """
        print(f"🔮 Starting nnU-Net inference...")
        print(f"   Input: {input_folder}")
        print(f"   Output: {output_folder}")
        
        # Create output directory
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Verify environment
        if 'nnUNet_results' not in os.environ:
            raise EnvironmentError("nnUNet_results environment variable not set")
        
        # Extract dataset number
        dataset_num = dataset_id.split('_')[0].replace('Dataset', '')
        
        # Construct model path
        model_folder = Path(os.environ['nnUNet_results']) / dataset_id / \
                      f'nnUNetTrainer__{configuration}' / f'fold_{fold}'
        
        if not model_folder.exists():
            raise ValueError(f"Model not found at {model_folder}")
        
        print(f"✓ Using model: {model_folder}")
        
        # Build inference command
        infer_cmd = [
            'nnUNetv2_predict',
            '-i', input_folder,
            '-o', output_folder,
            '-d', dataset_num,
            '-c', configuration,
            '-f', fold,
            '-chk', checkpoint
        ]
        
        if use_tta:
            infer_cmd.extend(['--disable_tta', 'false'])
        
        # Run inference
        print("🔄 Running inference...")
        try:
            result = subprocess.run(
                infer_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ Inference complete")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Inference failed: {e.stderr}")
        
        # Count output files
        output_files = list(Path(output_folder).glob('*.nii.gz'))
        
        summary = f"""
nnU-Net Inference Complete
{'='*60}

Configuration:
- Dataset: {dataset_id}
- Configuration: {configuration}
- Fold: {fold}
- Checkpoint: {checkpoint}
- Test-time augmentation: {use_tta}

Results:
- Segmented {len(output_files)} images
- Output location: {output_folder}

Segmentation masks saved in NIfTI format.
"""
        
        return summary.strip()