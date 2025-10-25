"""
TotalSegmentator Tool
Automated anatomical structure segmentation
"""

from smolagents import Tool
from totalsegmentator.python_api import totalsegmentator
from pathlib import Path
from typing import Optional, List
import SimpleITK as sitk
import numpy as np


class TotalSegmentatorTool(Tool):
    """
    Tool for TotalSegmentator anatomical segmentation
    Supports 100+ structures in CT and 50+ in MRI
    """
    
    name = "totalsegmentator"
    description = """Segments anatomical structures using TotalSegmentator.
    
    Inputs:
    - input_path: Path to CT/MRI scan (NIfTI or DICOM)
    - output_path: Path to save segmentation masks
    - task: Segmentation task ('total', 'total_mr', 'lung_vessels', 'body', etc.)
    - roi_subset: List of specific organs to segment (optional)
    - fast: Use fast mode (default: False)
    - ml: Generate multilabel mask (default: True)
    - device: Device to use ('gpu', 'cpu', 'mps')
    
    Outputs:
    - Segmentation masks (NIfTI format)
    - Statistics file (optional)
    """
    
    inputs = {
        "input_path": {"type": "string", "description": "Input scan path"},
        "output_path": {"type": "string", "description": "Output directory"},
        "task": {"type": "string", "description": "Segmentation task", "default": "total"},
        "roi_subset": {"type": "array", "description": "Specific organs", "nullable": True},
        "fast": {"type": "boolean", "description": "Fast mode", "default": False},
        "ml": {"type": "boolean", "description": "Multilabel output", "default": True},
        "device": {"type": "string", "description": "Device", "default": "gpu"}
    }
    
    output_type = "string"
    
    def forward(self, input_path: str, output_path: str, task: str = "total",
                roi_subset: Optional[List[str]] = None, fast: bool = False,
                ml: bool = True, device: str = "gpu") -> str:
        """
        Segment anatomical structures
        
        Args:
            input_path: Path to input scan
            output_path: Output directory path
            task: Segmentation task type
            roi_subset: List of specific organs/structures
            fast: Use fast inference mode
            ml: Generate multilabel mask
            device: Computation device
            
        Returns:
            Segmentation summary
        """
        print(f"🏥 Starting TotalSegmentator...")
        print(f"   Input: {input_path}")
        print(f"   Task: {task}")
        
        # Validate input
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory: {output_dir}")
        
        # Run TotalSegmentator
        print("🔄 Segmenting...")
        
        try:
            totalsegmentator(
                input=str(input_file),
                output=str(output_dir),
                task=task,
                roi_subset=roi_subset,
                ml=ml,
                nr_thr_saving=1,
                fast=fast,
                device=device,
                quiet=False
            )
            
            print("✓ Segmentation complete")
            
        except Exception as e:
            raise RuntimeError(f"TotalSegmentator failed: {str(e)}")
        
        # Count segmented structures
        output_files = list(output_dir.glob('*.nii.gz'))
        
        if roi_subset:
            segmented = roi_subset
        else:
            segmented = [f.stem.replace('.nii', '') for f in output_files]
        
        # Calculate volumes if requested
        volumes = {}
        if ml and (output_dir / 'segmentations.nii.gz').exists():
            print("📊 Calculating volumes...")
            seg_img = sitk.ReadImage(str(output_dir / 'segmentations.nii.gz'))
            seg_array = sitk.GetArrayFromImage(seg_img)
            spacing = seg_img.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
            
            unique_labels = np.unique(seg_array)
            for label in unique_labels:
                if label > 0:
                    n_voxels = np.sum(seg_array == label)
                    volume_mm3 = n_voxels * voxel_volume
                    volume_cm3 = volume_mm3 / 1000
                    volumes[int(label)] = {
                        'voxels': int(n_voxels),
                        'volume_mm3': float(volume_mm3),
                        'volume_cm3': float(volume_cm3)
                    }
        
        # Create summary
        summary = f"""
TotalSegmentator Complete
{'='*60}

Configuration:
- Task: {task}
- Input: {input_file.name}
- Fast mode: {fast}
- Device: {device}

Results:
- Total structures segmented: {len(segmented)}
- Output files: {len(output_files)}

Segmented Structures:
"""
        for struct in sorted(segmented[:20]):  # Show first 20
            summary += f"  - {struct}\n"
        
        if len(segmented) > 20:
            summary += f"  ... and {len(segmented) - 20} more\n"
        
        if volumes:
            summary += f"\nVolume Statistics (top 5):\n"
            sorted_volumes = sorted(volumes.items(), 
                                   key=lambda x: x[1]['volume_cm3'], 
                                   reverse=True)[:5]
            for label, vol in sorted_volumes:
                summary += f"  Label {label}: {vol['volume_cm3']:.2f} cm³\n"
        
        summary += f"\nAll masks saved to: {output_dir}\n"
        
        return summary.strip()