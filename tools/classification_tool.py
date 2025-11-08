"""
PyCaret Classification Tool
Automated ML for classification tasks
"""

from smolagents import Tool
import pandas as pd
from pathlib import Path
from pycaret.classification import (
    setup, compare_models, tune_model, blend_models,
    save_model, plot_model, pull
)
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class PyCaretClassificationTool(Tool):
    """
    Tool for automated classification model training using PyCaret
    """
    
    name = "pycaret_classification_training"
    description = """Trains classification models using PyCaret AutoML.
    
    Inputs:
    - input_file: Path to training data (CSV/Excel)
    - target_column: Target variable name
    - output_dir: Directory to save models and results
    - exclude_models: List of models to exclude (optional)
    - n_select: Number of top models to tune and blend (default: 3)
    - fold: Number of CV folds (default: 10)
    - normalize: Apply normalization (default: False)
    - transformation: Apply transformations (default: False)
    
    Outputs:
    - Trained models (tuned + blended)
    - Performance metrics
    - Evaluation plots
    - Model comparison report
    """
    
    inputs = {
        "input_file": {"type": "string", "description": "Path to training data"},
        "target_column": {"type": "string", "description": "Target column name"},
        "output_dir": {"type": "string", "description": "Output directory"},
        "exclude_models": {"type": "array", "description": "Models to exclude", "nullable": True},
        "n_select": {"type": "integer", "description": "Top N models to select", "default": 3},
        "fold": {"type": "integer", "description": "CV folds", "default": 10},
        "normalize": {"type": "boolean", "description": "Normalize features", "default": False},
        "transformation": {"type": "boolean", "description": "Transform features", "default": False}
    }
    
    output_type = "string"
    
    def forward(self, input_file: str, target_column: str, output_dir: str,
                exclude_models: Optional[List[str]] = None, n_select: int = 3,
                fold: int = 10, normalize: bool = False, 
                transformation: bool = False) -> str:
        """
        Train classification models
        
        Args:
            input_file: Path to training data
            target_column: Name of target column
            output_dir: Output directory for results
            exclude_models: List of model names to exclude
            n_select: Number of top models to select
            fold: Number of cross-validation folds
            normalize: Whether to normalize features
            transformation: Whether to apply transformations
            
        Returns:
            Summary report string
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)
        plots_dir = output_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"📊 Loading training data from {input_file}...")
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        else:
            data = pd.read_excel(input_file)
        
        print(f"✓ Loaded {len(data)} samples with {len(data.columns)} features")
        
        # Setup PyCaret
        print("🔧 Setting up PyCaret environment...")
        exp = setup(
            data=data,
            target=target_column,
            fold=fold,
            session_id=42,
            normalize=normalize,
            transformation=transformation,
            verbose=False,
            html=False
        )
        
        print("✓ PyCaret setup complete")
        
        # Compare models
        print("🔍 Comparing baseline models...")
        if exclude_models:
            best_models = compare_models(
                exclude=exclude_models,
                n_select=n_select,
                sort='AUC'
            )
        else:
            best_models = compare_models(
                n_select=n_select,
                sort='AUC'
            )
        
        # Get comparison results
        comparison_results = pull()
        comparison_results.to_csv(output_path / 'model_comparison.csv', index=False)
        print(f"✓ Compared models. Top {n_select} selected.")
        
        # Tune selected models
        print(f"⚙️ Tuning top {n_select} models...")
        tuned_models = []
        for idx, model in enumerate(best_models if isinstance(best_models, list) else [best_models]):
            print(f"  Tuning model {idx+1}/{n_select}...")
            tuned = tune_model(model, optimize='AUC')
            tuned_models.append(tuned)
            
            # Save tuned model
            model_path = models_dir / f'tuned_model_{idx+1}'
            save_model(tuned, str(model_path))
            print(f"  ✓ Saved to {model_path.name}")
        
        # Create blended model
        print("🔀 Creating blended ensemble...")
        blended = blend_models(tuned_models)
        blended_path = models_dir / 'blended_model'
        save_model(blended, str(blended_path))
        print(f"✓ Blended model saved")
        
        # Get final metrics
        final_metrics = pull()
        final_metrics.to_csv(output_path / 'final_metrics.csv', index=False)
        
        # Generate evaluation plots
        print("📊 Generating evaluation plots...")
        plot_types = ['auc', 'confusion_matrix', 'pr', 'class_report', 
                     'boundary', 'learning', 'calibration', 'feature']
        
        for plot_type in plot_types:
            try:
                plot_model(blended, plot=plot_type, save=True)
                # Move to plots directory
                import shutil
                import glob
                for f in glob.glob(f'{plot_type}.png'):
                    shutil.move(f, plots_dir / f'{plot_type}.png')
            except:
                continue
        
        print("✓ Plots generated")
        
        # Create summary
        summary = self._create_summary(comparison_results, final_metrics, 
                                      n_select, output_path)
        
        # Save summary
        with open(output_path / 'training_summary.txt', 'w') as f:
            f.write(summary)
        
        return summary
    
    def _create_summary(self, comparison_df: pd.DataFrame, 
                       metrics_df: pd.DataFrame, n_select: int, 
                       output_path: Path) -> str:
        """Create training summary report"""
        
        summary = f"""
Classification Model Training Complete
{'='*60}

Models Trained and Evaluated:
- Total models compared: {len(comparison_df)}
- Top models selected: {n_select}
- Models tuned: {n_select}
- Final blended ensemble created

Blended Model Performance (Cross-Validation):
"""
        # Extract key metrics from final results
        if not metrics_df.empty:
            for col in ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1']:
                if col in metrics_df.columns:
                    mean_val = metrics_df[col].iloc[0]
                    summary += f"  {col}: {mean_val:.4f}\n"
        
        summary += f"""
Output Files:
- Model comparison: model_comparison.csv
- Final metrics: final_metrics.csv
- Tuned models: models/tuned_model_*.pkl
- Blended model: models/blended_model.pkl
- Evaluation plots: plots/

All results saved to: {output_path}
"""
        return summary.strip()