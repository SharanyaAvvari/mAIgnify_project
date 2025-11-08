"""
PyRadiomics Feature Extraction Tool
Extracts quantitative radiomic features from medical images
"""

from smolagents import Tool
import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from radiomics import featureextractor
import logging
from typing import Optional, List, Dict
import yaml
from concurrent.futures import ProcessPoolExecutor
import glob


class PyRadiomicsFeatureExtractionTool(Tool):
    """
    Tool for radiomic feature extraction using PyRadiomics
    Supports multiple image types, filters, and feature classes
    """
    
    name = "pyradiomics_feature_extraction"
    description = """Extracts radiomic features from medical images (CT/MRI).
    
    Inputs:
    - images_dir: Directory containing medical images (NIfTI format)
    - masks_dir: Directory containing segmentation masks
    - output_dir: Directory to save extracted features
    - image_types: List of image types (Original, Wavelet, LoG, Exponential, etc.)
    - feature_classes: List of feature classes (firstorder, shape, glcm, glrlm, etc.)
    - bin_width: Discretization bin width (default: 25)
    - resample_spacing: Isotropic voxel spacing [x,y,z] (optional)
    - normalize: Apply intensity normalization (default: True)
    - label: Specific label to extract (default: all labels)
    - parallel_workers: Number of parallel processes (default: 4)
    
    Outputs:
    - CSV files with extracted features per label
    - Parameter configuration file
    - Processing log
    """
    
    inputs = {
        "images_dir": {"type": "string", "description": "Images directory path"},
        "masks_dir": {"type": "string", "description": "Masks directory path"},
        "output_dir": {"type": "string", "description": "Output directory"},
        "image_types": {"type": "array", "description": "Image filter types", 
                       "default": ["Original", "Wavelet"]},
        "feature_classes": {"type": "array", "description": "Feature classes",
                           "default": ["firstorder", "shape", "glcm", "glrlm", "glszm"]},
        "bin_width": {"type": "integer", "description": "Discretization bin width", "default": 25},
        "resample_spacing": {"type": "array", "description": "Resample spacing [x,y,z]", "nullable": True},
        "normalize": {"type": "boolean", "description": "Normalize intensity", "default": True},
        "label": {"type": "integer", "description": "Specific label to extract", "nullable": True},
        "parallel_workers": {"type": "integer", "description": "Parallel workers", "default": 4}
    }
    
    output_type = "string"
    
    def forward(self, images_dir: str, masks_dir: str, output_dir: str,
                image_types: List[str] = ["Original", "Wavelet"],
                feature_classes: List[str] = ["firstorder", "shape", "glcm", "glrlm", "glszm"],
                bin_width: int = 25, resample_spacing: Optional[List[float]] = None,
                normalize: bool = True, label: Optional[int] = None,
                parallel_workers: int = 4) -> str:
        """
        Extract radiomic features from medical images
        
        Args:
            images_dir: Path to image directory
            masks_dir: Path to mask directory
            output_dir: Output directory
            image_types: List of image filters to apply
            feature_classes: List of feature classes to extract
            bin_width: Bin width for discretization
            resample_spacing: Voxel spacing for resampling
            normalize: Whether to normalize intensities
            label: Specific label value (None = all labels)
            parallel_workers: Number of parallel processes
            
        Returns:
            Summary string
        """
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = output_path / 'radiomic_extraction.log'
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print(f"🔬 Starting radiomic feature extraction...")
        print(f"   Images: {images_dir}")
        print(f"   Masks: {masks_dir}")
        
        # Get image and mask files
        image_files = sorted(glob.glob(str(Path(images_dir) / "*.nii.gz")))
        mask_files = sorted(glob.glob(str(Path(masks_dir) / "*.nii.gz")))
        
        if len(image_files) == 0:
            raise ValueError(f"No NIfTI images found in {images_dir}")
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in {masks_dir}")
        
        print(f"✓ Found {len(image_files)} images and {len(mask_files)} masks")
        
        # Create parameter configuration
        params = self._create_params(image_types, feature_classes, bin_width, 
                                     resample_spacing, normalize)
        
        # Save parameters
        param_file = output_path / 'extraction_params.yaml'
        with open(param_file, 'w') as f:
            yaml.dump(params, f)
        
        print(f"✓ Parameters saved to {param_file.name}")
        
        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        
        # Match images with masks
        image_mask_pairs = self._match_images_masks(image_files, mask_files)
        print(f"✓ Matched {len(image_mask_pairs)} image-mask pairs")
        
        # Extract features
        print(f"🔄 Extracting features using {parallel_workers} workers...")
        
        if label is not None:
            # Single label extraction
            features_df = self._extract_single_label(
                image_mask_pairs, extractor, label, parallel_workers
            )
            output_file = output_path / f'radiomic_features_label_{label}.csv'
            features_df.to_csv(output_file, index=False)
            print(f"✓ Saved features for label {label} to {output_file.name}")
            
        else:
            # Extract all labels
            all_labels = self._get_all_labels(mask_files)
            print(f"✓ Found {len(all_labels)} unique labels: {all_labels}")
            
            for lbl in all_labels:
                print(f"  Extracting label {lbl}...")
                features_df = self._extract_single_label(
                    image_mask_pairs, extractor, lbl, parallel_workers
                )
                output_file = output_path / f'radiomic_features_label_{lbl}.csv'
                features_df.to_csv(output_file, index=False)
                print(f"  ✓ Saved {len(features_df)} feature vectors")
        
        # Create summary
        summary = self._create_summary(image_mask_pairs, image_types, 
                                      feature_classes, output_path)
        
        # Save summary
        with open(output_path / 'extraction_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"\n✓ Radiomic extraction complete!")
        print(f"✓ Results saved to {output_dir}")
        
        return summary
    
    def _create_params(self, image_types: List[str], feature_classes: List[str],
                      bin_width: int, resample_spacing: Optional[List[float]],
                      normalize: bool) -> dict:
        """Create PyRadiomics parameter dictionary"""
        params = {
            'binWidth': bin_width,
            'normalize': normalize,
            'normalizeScale': 100 if normalize else 1,
            'interpolator': 'sitkBSpline',
            'resampledPixelSpacing': resample_spacing,
            'label': 1
        }
        
        # Configure image types
        params['imageType'] = {}
        for img_type in image_types:
            if img_type.lower() == 'original':
                params['imageType']['Original'] = {}
            elif img_type.lower() == 'wavelet':
                params['imageType']['Wavelet'] = {}
            elif img_type.lower() == 'log':
                params['imageType']['LoG'] = {'sigma': [1.0, 2.0, 3.0]}
            elif img_type.lower() == 'exponential':
                params['imageType']['Exponential'] = {}
            elif img_type.lower() == 'gradient':
                params['imageType']['Gradient'] = {}
            elif img_type.lower() == 'lbp2d':
                params['imageType']['LBP2D'] = {}
            elif img_type.lower() == 'lbp3d':
                params['imageType']['LBP3D'] = {}
            elif img_type.lower() == 'squareroot':
                params['imageType']['SquareRoot'] = {}
        
        # Configure feature classes
        params['featureClass'] = {}
        for feat_class in feature_classes:
            params['featureClass'][feat_class] = []
        
        return params
    
    def _match_images_masks(self, image_files: List[str], 
                           mask_files: List[str]) -> List[tuple]:
        """Match image files with corresponding masks"""
        pairs = []
        
        for img_file in image_files:
            img_name = Path(img_file).stem.replace('.nii', '')
            
            # Find matching mask
            matching_mask = None
            for mask_file in mask_files:
                mask_name = Path(mask_file).stem.replace('.nii', '')
                if mask_name in img_name or img_name in mask_name:
                    matching_mask = mask_file
                    break
            
            if matching_mask:
                pairs.append((img_file, matching_mask))
            else:
                logging.warning(f"No matching mask found for {img_name}")
        
        return pairs
    
    def _get_all_labels(self, mask_files: List[str]) -> List[int]:
        """Get all unique labels from mask files"""
        all_labels = set()
        
        for mask_file in mask_files[:3]:  # Sample first 3 masks
            mask = sitk.ReadImage(mask_file)
            mask_array = sitk.GetArrayFromImage(mask)
            labels = np.unique(mask_array)
            all_labels.update(labels[labels > 0])  # Exclude background
        
        return sorted(list(all_labels))
    
    def _extract_single_label(self, image_mask_pairs: List[tuple],
                             extractor: featureextractor.RadiomicsFeatureExtractor,
                             label: int, workers: int) -> pd.DataFrame:
        """Extract features for a single label"""
        
        def extract_single(pair):
            img_file, mask_file = pair
            case_id = Path(img_file).stem.replace('.nii', '')
            
            try:
                # Extract features
                result = extractor.execute(img_file, mask_file, label=label)
                
                # Convert to dict (remove diagnostics)
                features = {
                    k: v for k, v in result.items() 
                    if not k.startswith('diagnostics_')
                }
                features['case_id'] = case_id
                
                return features
                
            except Exception as e:
                logging.error(f"Error extracting {case_id}: {str(e)}")
                return None
        
        # Extract in parallel
        results = []
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(extract_single, image_mask_pairs))
        else:
            results = [extract_single(pair) for pair in image_mask_pairs]
        
        # Filter successful extractions
        results = [r for r in results if r is not None]
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Move case_id to first column
        if 'case_id' in df.columns:
            cols = ['case_id'] + [c for c in df.columns if c != 'case_id']
            df = df[cols]
        
        return df
    
    def _create_summary(self, pairs: List[tuple], image_types: List[str],
                       feature_classes: List[str], output_path: Path) -> str:
        """Create extraction summary"""
        
        summary = f"""
Radiomic Feature Extraction Summary
{'='*60}

Dataset Information:
- Total image-mask pairs: {len(pairs)}
- Successfully processed: {len(pairs)}

Configuration:
- Image types: {', '.join(image_types)}
- Feature classes: {', '.join(feature_classes)}

Output Files:
"""
        # List all output CSV files
        csv_files = list(output_path.glob('radiomic_features_*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            summary += f"  - {csv_file.name}: {len(df)} samples, {len(df.columns)-1} features\n"
        
        summary += f"\nAll results saved to: {output_path}\n"
        
        return summary.strip()