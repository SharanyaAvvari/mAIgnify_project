"""
nnU-Net Training Tool
Automates nnU-Net segmentation model training
"""

from smolagents import Tool
import subprocess
import os
from pathlib import Path
from typing import Optional
import json


class NNUNetTrainingTool(Tool):
    """
    Tool for training nnU-Net segmentation models
    """
    
    name = "nnunet_training"
    description = """Trains nnU-Net segmentation models on medical imaging datasets.
    
    Inputs:
    - dataset_id: nnU-Net dataset ID (e.g., 'Dataset001_BrainTumor')
    - configuration: Model configuration ('2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres')
    - fold: Fold number ('all', '0', '1', '2', '3', '4')
    - trainer: Trainer class name (default: 'nnUNetTrainer')
    - num_gpus: Number of GPUs to use (default: 1)
    - continue_training: Continue from checkpoint (default: False)
    
    Outputs:
    - Trained model checkpoints
    - Validation metrics
    - Training logs
    """
    
    inputs = {
        "dataset_id": {"type": "string", "description": "Dataset ID (e.g., Dataset001_BrainTumor)"},
        "configuration": {"type": "string", "description": "Model config", "default": "3d_fullres"},
        "fold": {"type": "string", "description": "Fold number", "default": "all"},
        "trainer": {"type": "string", "description": "Trainer class", "default": "nnUNetTrainer"},
        "num_gpus": {"type": "integer", "description": "Number of GPUs", "default": 1},
        "continue_training": {"type": "boolean", "description": "Continue training", "default": False}
    }
    
    output_type = "string"
    
    def forward(self, dataset_id: str, configuration: str = "3d_fullres",
                fold: str = "all", trainer: str = "nnUNetTrainer",
                num_gpus: int = 1, continue_training: bool = False) -> str:
        """
        Train nnU-Net model
        
        Args:
            dataset_id: nnU-Net dataset identifier
            configuration: Model configuration type
            fold: Cross-validation fold
            trainer: Trainer class name
            num_gpus: Number of GPUs
            continue_training: Whether to continue from checkpoint
            
        Returns:
            Training summary
        """
        print(f"🧠 Starting nnU-Net training...")
        print(f"   Dataset: {dataset_id}")
        print(f"   Configuration: {configuration}")
        print(f"   Fold: {fold}")
        
        # Verify environment variables
        required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
        for var in required_vars:
            if var not in os.environ:
                raise EnvironmentError(
                    f"Environment variable {var} not set. "
                    f"Please configure nnU-Net environment."
                )
        
        # Extract dataset number
        dataset_num = dataset_id.split('_')[0].replace('Dataset', '')
        
        # Step 1: Preprocess (if not already done)
        print("📋 Step 1: Preprocessing dataset...")
        preprocess_cmd = [
            'nnUNetv2_plan_and_preprocess',
            '-d', dataset_num,
            '--verify_dataset_integrity'
        ]
        
        try:
            result = subprocess.run(
                preprocess_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ Preprocessing complete")
        except subprocess.CalledProcessError as e:
            if "already" in e.stdout.lower():
                print("✓ Dataset already preprocessed")
            else:
                raise RuntimeError(f"Preprocessing failed: {e.stderr}")
        
        # Step 2: Train model
        print(f"🔥 Step 2: Training {configuration} model...")
        
        train_cmd = [
            'nnUNetv2_train',
            dataset_num,
            configuration,
            fold,
            '-tr', trainer
        ]
        
        if continue_training:
            train_cmd.append('--c')
        
        if num_gpus > 1:
            train_cmd.extend(['-num_gpus', str(num_gpus)])
        
        # Run training
        try:
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ Training complete")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Training failed: {e.stderr}")
        
        # Get results path
        results_path = Path(os.environ['nnUNet_results']) / dataset_id / \
                      f'nnUNetTrainer__{configuration}' / f'fold_{fold}'
        
        # Read validation metrics if available
        metrics_file = results_path / 'validation_raw' / 'summary.json'
        metrics_summary = "Metrics file not found"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            metrics_summary = f"""
Validation Metrics:
- Mean Dice: {metrics.get('mean_dice', 'N/A')}
- Mean IoU: {metrics.get('mean_iou', 'N/A')}
"""
        
        summary = f"""
nnU-Net Training Complete
{'='*60}

Configuration:
- Dataset: {dataset_id}
- Configuration: {configuration}
- Fold: {fold}
- Trainer: {trainer}

{metrics_summary}

Model Location:
{results_path}

Checkpoints saved:
- checkpoint_best.pth
- checkpoint_final.pth
"""
        
        return summary.strip()