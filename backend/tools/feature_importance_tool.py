"""
Feature Importance Analysis Tool
Multiple feature selection strategies
"""

from smolagents import Tool
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, 
    mutual_info_classif, mutual_info_regression,
    RFE
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


class FeatureImportanceAnalysisTool(Tool):
    """
    Tool for feature importance analysis and selection
    Supports multiple methods: Random Forest, ANOVA, Mutual Information, RFE
    """
    
    name = "feature_importance_analysis"
    description = """Performs feature importance analysis and selection.
    
    Inputs:
    - input_file: Path to CSV/Excel file
    - target_column: Name of target variable
    - output_dir: Directory to save results
    - method: Feature selection method (random_forest, anova, mutual_info, rfe)
    - task_type: Classification or regression (auto-detected if not specified)
    - top_k_features: List of k values for top-k feature selection [5, 10, 20]
    
    Outputs:
    - CSV files with selected features
    - Feature importance plots
    - Feature correlation analysis
    """
    
    inputs = {
        "input_file": {"type": "string", "description": "Path to data file"},
        "target_column": {"type": "string", "description": "Target variable name"},
        "output_dir": {"type": "string", "description": "Output directory"},
        "method": {"type": "string", "description": "Selection method", "default": "random_forest"},
        "task_type": {"type": "string", "description": "classification or regression", "nullable": True},
        "top_k_features": {"type": "array", "description": "List of k values", "default": [5, 10, 20]}
    }
    
    output_type = "string"
    
    def forward(self, input_file: str, target_column: str, output_dir: str,
                method: str = "random_forest", task_type: Optional[str] = None,
                top_k_features: List[int] = [5, 10, 20]) -> str:
        """
        Execute feature importance analysis
        
        Args:
            input_file: Path to input data
            target_column: Target variable column name
            output_dir: Output directory
            method: Feature selection method
            task_type: Task type (auto-detected if None)
            top_k_features: List of k values for feature selection
            
        Returns:
            Summary string
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"📊 Loading data from {input_file}...")
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Auto-detect task type
        if task_type is None:
            task_type = self._detect_task_type(y)
        
        print(f"✓ Task type: {task_type}")
        print(f"✓ Features: {len(X.columns)}, Samples: {len(X)}")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical variables
        X_encoded, encoders = self._encode_features(X)
        
        # Encode target if classification
        if task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Calculate feature importance
        print(f"🔍 Calculating feature importance using {method}...")
        importance_scores = self._calculate_importance(
            X_encoded, y, method, task_type
        )
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Save full importance scores
        importance_df.to_csv(output_path / 'feature_importance_scores.csv', index=False)
        
        # Save top-k feature sets
        saved_files = []
        for k in top_k_features:
            k_actual = min(k, len(importance_df))
            top_features = importance_df.head(k_actual)['feature'].tolist()
            
            # Create subset with top features + target
            subset_df = df[top_features + [target_column]]
            output_file = output_path / f'top_{k_actual}_features.csv'
            subset_df.to_csv(output_file, index=False)
            saved_files.append(str(output_file))
            
            print(f"✓ Saved top {k_actual} features to {output_file.name}")
        
        # Generate visualizations
        self._create_visualizations(importance_df, X_encoded, output_path)
        
        # Create summary
        summary = self._create_summary(importance_df, top_k_features, saved_files)
        
        return summary
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect if task is classification or regression"""
        if y.dtype == 'object' or y.nunique() < 20:
            return 'classification'
        else:
            return 'regression'
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Numeric: fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Categorical: fill with mode
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')
        
        return X
    
    def _encode_features(self, X: pd.DataFrame) -> tuple:
        """Encode categorical features"""
        X_encoded = X.copy()
        encoders = {}
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        
        return X_encoded, encoders
    
    def _calculate_importance(self, X: pd.DataFrame, y: np.ndarray, 
                             method: str, task_type: str) -> np.ndarray:
        """Calculate feature importance using specified method"""
        
        if method == 'random_forest':
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            return model.feature_importances_
        
        elif method == 'anova':
            if task_type == 'classification':
                selector = SelectKBest(f_classif, k='all')
            else:
                selector = SelectKBest(f_regression, k='all')
            
            selector.fit(X, y)
            return selector.scores_
        
        elif method == 'mutual_info':
            if task_type == 'classification':
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            
            return scores
        
        elif method == 'rfe':
            if task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            rfe = RFE(estimator, n_features_to_select=len(X.columns))
            rfe.fit(X, y)
            return rfe.ranking_
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_visualizations(self, importance_df: pd.DataFrame, 
                              X: pd.DataFrame, output_path: Path):
        """Create feature importance visualizations"""
        
        # 1. Feature Importance Bar Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        top_20 = importance_df.head(20)
        ax.barh(range(len(top_20)), top_20['importance'])
        ax.set_yticks(range(len(top_20)))
        ax.set_yticklabels(top_20['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 20 Feature Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cumulative Importance
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Feature Importance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'cumulative_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Correlation Heatmap (top 20)
        top_features = importance_df.head(20)['feature'].tolist()
        corr_matrix = X[top_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title('Correlation Heatmap - Top 20 Features')
        plt.tight_layout()
        plt.savefig(output_path / 'top_features_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary(self, importance_df: pd.DataFrame, 
                       top_k_features: List[int], saved_files: List[str]) -> str:
        """Create summary report"""
        
        top_5 = importance_df.head(5)
        
        summary = f"""
Feature Importance Analysis Complete
{'='*50}

Total Features Analyzed: {len(importance_df)}

Top 5 Most Important Features:
"""
        for idx, row in top_5.iterrows():
            summary += f"  {idx+1}. {row['feature']}: {row['importance']:.4f}\n"
        
        summary += f"\nFeature Sets Saved:\n"
        for f in saved_files:
            summary += f"  - {Path(f).name}\n"
        
        summary += f"\nVisualizations created:\n"
        summary += "  - feature_importance_plot.png\n"
        summary += "  - cumulative_importance.png\n"
        summary += "  - top_features_correlation.png\n"
        
        return summary.strip()