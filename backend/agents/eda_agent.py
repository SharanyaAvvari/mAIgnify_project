"""
EDA Agent - Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

class EDAAgent:
    def __init__(self):
        self.name = "EDA Agent"
        
    async def execute(self, prompt: str, file_paths: List[str], output_dir: str) -> dict:
        """
        Perform comprehensive EDA on dataset
        """
        results = {}
        
        try:
            # Load dataset
            data = None
            for file_path in file_paths:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                    break
                elif file_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path)
                    break
            
            if data is None:
                return {'summary': 'No CSV/Excel file found', 'error': 'No tabular data provided'}
            
            # Basic statistics
            stats_summary = data.describe().to_dict()
            
            # Missing values
            missing_data = data.isnull().sum().to_dict()
            
            # Data types
            dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
            
            # Generate visualizations
            self._create_visualizations(data, output_dir)
            
            # Create summary report
            report_path = os.path.join(output_dir, 'eda_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"=== EDA REPORT ===\n\n")
                f.write(f"Dataset Shape: {data.shape[0]} rows x {data.shape[1]} columns\n\n")
                f.write(f"Columns: {', '.join(data.columns)}\n\n")
                f.write(f"Missing Values:\n{pd.Series(missing_data)}\n\n")
                f.write(f"Summary Statistics:\n{data.describe()}\n")
            
            results = {
                'summary': f'EDA completed on dataset with {data.shape[0]} rows and {data.shape[1]} columns',
                'shape': data.shape,
                'columns': list(data.columns),
                'missing_values': missing_data,
                'statistics': stats_summary,
                'visualizations_created': True
            }
            
        except Exception as e:
            results = {
                'summary': f'EDA failed: {str(e)}',
                'error': str(e)
            }
        
        return results
    
    def _create_visualizations(self, data: pd.DataFrame, output_dir: str):
        """Create EDA visualizations"""
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Distribution plots for numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(len(numeric_cols), 4)
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols[:n_cols]):
                data[col].hist(ax=axes[idx], bins=30, edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation = data[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Missing data visualization
        missing = data.isnull().sum()
        if missing.sum() > 0:
            plt.figure(figsize=(10, 6))
            missing[missing > 0].sort_values(ascending=False).plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.xlabel('Column')
            plt.ylabel('Number of Missing Values')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=150, bbox_inches='tight')
            plt.close()