"""
backend/tools/pycaret_inference_tool.py

PyCaret Inference Tool
Run predictions with trained PyCaret classification models
"""

from smolagents import Tool
import pandas as pd
import logging
from pathlib import Path
from pycaret.classification import load_model, predict_model


class PyCaretInferenceTool(Tool):
    """
    Tool for running inference with trained PyCaret classification models.
    """
    
    name = "pycaret_inference"
    description = """
    Runs inference on new data using trained PyCaret classification models.
    
    Features:
    - Load saved PyCaret models
    - Predict on new data
    - Generate prediction probabilities
    - Export results to CSV
    
    Parameters:
    - model_path: Path to saved PyCaret model (.pkl)
    - test_data: Path to CSV file with test data
    - output_file: Path to save predictions
    """
    
    inputs = {
        "model_path": {
            "type": "string",
            "description": "Path to trained model"
        },
        "test_data": {
            "type": "string",
            "description": "Test data CSV file"
        },
        "output_file": {
            "type": "string",
            "description": "Output predictions file"
        }
    }
    
    output_type = "string"
    
    def _init_(self):
        super()._init_()
        self.logger = logging.getLogger(_name_)
    
    def forward(self,
                model_path: str,
                test_data: str,
                output_file: str) -> str:
        """
        Run inference with PyCaret model.
        
        Returns:
            Path to predictions CSV file
        """
        
        # Load model (remove .pkl extension if present)
        model_path_str = str(Path(model_path).with_suffix(''))
        model = load_model(model_path_str)
        
        # Load test data
        df_test = pd.read_csv(test_data)
        self.logger.info(f"Loaded test data: {df_test.shape[0]} rows")
        
        # Make predictions
        predictions = predict_model(model, data=df_test)
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        
        self.logger.info(f"Predictions saved to {output_path}")
        
        return str(output_path)