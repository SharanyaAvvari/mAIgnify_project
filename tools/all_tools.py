import os
import subprocess
from pathlib import Path
import nibabel as nib
import zipfile


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_totalsegmentator(input_path: str, outdir: str):
"""Call TotalSegmentator python API. Returns path to directory with masks."""
from totalsegmentator.python_api import totalsegmentator
totalsegmentator(input_path=input_path, output_path=outdir, task="total")
return outdir




def run_nnunet_predictor(input_path: str, model_folder: str, outdir: str):
"""Placeholder: call nnUNet prediction. Adjust for your nnU-Net installation.
Could be a shell call to nnUNetv2 or call into nnunetv2 API.
"""
# Example using a shell command (if nnUNet CLI installed):
cmd = [
"nnUNetv2_predict", # replace with your nnU-Net CLI command
"--input", input_path,
"--output_folder", outdir,
"--model", model_folder
]
try:
subprocess.check_call(cmd)
except Exception as e:
raise RuntimeError(f"nnU-Net call failed: {e}")
return outdir




def extract_radiomics_features(image_path: str, mask_path: str):
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor()
result = extractor.execute(image_path, mask_path)
return dict(result)




def zip_folder(folder_path: str, zip_path: str):
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
for root, dirs, files in os.walk(folder_path):
for f in files:
full = os.path.join(root, f)
rel = os.path.relpath(full, folder_path)
zf.write(full, arcname=rel)
return zip_path