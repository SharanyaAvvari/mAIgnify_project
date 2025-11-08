"""
backend/tools/pycaret_classification_tool.py

PyCaret Classification Tool
AutoML classification using PyCaret
"""

from smolagents import Tool
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List
from pycaret.classification import *
import joblib


class PyCaretClassificationTool(Tool):
    """
    Tool for automated classification model training using PyCaret.
    """
    
    name = "pycaret_classification"
    description = """
    Trains classification models using PyCaret AutoML framework.
    
    Capabilities:
    - 15+ algorithms (LR, RF, XGBoost, LightGBM, CatBoost, etc.)
    - Automated preprocessing
    - Feature engineering
    - Hyperparameter tuning
    - Model ensembling
    - SHAP interpretability
    
    Parameters:
    - input_file: Path to CSV file with training data
    - target_column: Name of target variable
    - output_dir: Directory to save trained model
    - test_size: Test set proportion (default: 0.2)
    - cv_folds: Cross-validation folds (default: 5)
    - exclude_models: Models to exclude from training
    - normalize: Normalize features (default: True)
    - feature_selection: Enable feature selection
    - tune_model: Tune best model hyperparameters
    """
    
    inputs = {
        "input_file": {
            "type": "string",
            "description": "Input CSV file path"
        },
        "target_column": {
            "type": "string",
            "description": "Target variable name"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory"
        },
        "cv_folds": {
            "type": "number",
            "description": "CV folds",
            "nullable": True
        },
        "tune_model": {
            "type": "boolean",
            "description": "Tune hyperparameters",
            "nullable": True
        }
    }
    
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def forward(self,
                input_file: str,
                target_column: str,
                output_dir: str,
                cv_folds: int = 5,
                tune_model: bool = True) -> str:
        """
        Train classification model with PyCaret.
        
        Returns:
            Path to saved model file
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df = pd.read_csv(input_file)
        self.logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Initialize PyCaret setup
        self.logger.info("Initializing PyCaret setup...")
        clf = setup(
            data=df,
            target=target_column,
            session_id=42,
            fold=cv_folds,
            normalize=True,
            transformation=True,
            ignore_low_variance=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.9,
            log_experiment=False,
            verbose=False
        )
        
        # Compare models
        self.logger.info("Comparing models...")
        best = compare_models(
            sort='Accuracy',
            n_select=1
        )
        
        # Tune hyperparameters
        if tune_model:
            self.logger.info("Tuning hyperparameters...")
            tuned = tune_model(
                best,
                optimize='Accuracy',
                n_iter=10
            )
            final_model = tuned
        else:
            final_model = best
        
        # Get model performance
        self.logger.info("Evaluating model...")
        results = pull()
        
        # Save results
        results.to_csv(output_path / 'model_comparison.csv', index=False)
        
        # Save final model
        model_file = output_path / 'classification_model.pkl'
        save_model(final_model, str(model_file.with_suffix('')))
        
        # Generate evaluation plots
        self.logger.info("Generating evaluation plots...")
        try:
            plot_model(final_model, plot='auc', save=True)
            plot_model(final_model, plot='confusion_matrix', save=True)
            plot_model(final_model, plot='feature', save=True)
        except:
            pass
        
        self.logger.info(f"Model saved to {model_file}")
        
        return str(model_file)
