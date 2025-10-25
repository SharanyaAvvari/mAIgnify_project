"""
PyCaret Regression Inference Tool
"""

from smolagents import Tool
import pandas as pd
from pathlib import Path
from pycaret.regression import load_model, predict_model
import warnings
warnings.filterwarnings('ignore')

class PyCaretRegressionInferenceTool(Tool):
    """
    Tool for making predictions with a trained PyCaret regression model.
    """
    
    name = "pycaret_regression_inference"
    description = """Loads a saved PyCaret regression model and predicts on new data.
    
    Inputs:
    - model_path: Path to the saved model file (without the .pkl extension).
    - input_file: Path to the data for prediction (CSV/Excel).
    - output_file: Path to save the prediction results (CSV).
    
    Outputs:
    - A CSV file with an added 'prediction_label' column.
    """
    
    inputs = {
        "model_path": {"type": "string", "description": "Path to the saved PyCaret model (e.g., 'path/to/my_model')"},
        "input_file": {"type": "string", "description": "Path to the input data file for prediction"},
        "output_file": {"type": "string", "description": "Path to save the output CSV with predictions"}
    }
    
    output_type = "string"
    
    def forward(self, model_path: str, input_file: str, output_file: str) -> str:
        """
        Performs inference using a saved PyCaret model.
        
        Args:
            model_path: The base path of the saved model.
            input_file: Path to the data for prediction.
            output_file: Path to save the prediction results.
            
        Returns:
            A summary string of the operation.
        """
        try:
            print(f"🔧 Loading model from {model_path}...")
            pipeline = load_model(model_path)
            print("✓ Model loaded successfully.")
            
            print(f"📊 Loading data for inference from {input_file}...")
            if input_file.endswith('.csv'):
                data = pd.read_csv(input_file)
            else:
                data = pd.read_excel(input_file)
            print(f"✓ Loaded {len(data)} records.")
            
            print("🧠 Making predictions...")
            predictions = predict_model(pipeline, data=data)
            # The prediction column is named 'prediction_label' by PyCaret
            print("✓ Predictions complete.")
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            predictions.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to {output_file}")
            
            return f"Inference complete. Predictions for {len(data)} records saved to {output_file}."

        except Exception as e:
            return f"An error occurred during inference: {e}"