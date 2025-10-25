"""
Exploratory Data Analysis Tool
Automated profiling, visualization, and statistical analysis
"""

from smolagents import Tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import json


class ExploratoryDataAnalysisTool(Tool):
    """
    Tool for comprehensive exploratory data analysis
    Generates statistics, visualizations, and reports
    """
    
    name = "exploratory_data_analysis"
    description = """Performs comprehensive exploratory data analysis on tabular datasets.
    
    Inputs:
    - input_file: Path to CSV or Excel file
    - output_dir: Directory to save results
    - sheet_name: (Optional) Sheet name for Excel files
    - create_plots: Whether to generate visualizations (default: True)
    
    Outputs:
    - Summary statistics
    - Visualizations (histograms, boxplots, correlation heatmaps, etc.)
    - Missing data analysis
    - Data quality report
    """
    
    inputs = {
        "input_file": {"type": "string", "description": "Path to data file (CSV/Excel)"},
        "output_dir": {"type": "string", "description": "Output directory path"},
        "sheet_name": {"type": "string", "description": "Excel sheet name", "nullable": True},
        "create_plots": {"type": "boolean", "description": "Generate visualizations", "default": True}
    }
    
    output_type = "string"
    
    def forward(self, input_file: str, output_dir: str, 
                sheet_name: Optional[str] = None, create_plots: bool = True) -> str:
        """
        Execute exploratory data analysis
        
        Args:
            input_file: Path to input data file
            output_dir: Directory to save outputs
            sheet_name: Excel sheet name (if applicable)
            create_plots: Whether to create visualizations
            
        Returns:
            Summary report as string
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"📊 Loading data from {input_file}...")
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file, sheet_name=sheet_name)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Generate analysis
        analysis_results = {}
        
        # 1. Basic Info
        analysis_results['basic_info'] = self._basic_info(df)
        
        # 2. Summary Statistics
        analysis_results['statistics'] = self._summary_statistics(df, output_path)
        
        # 3. Missing Data Analysis
        analysis_results['missing_data'] = self._missing_data_analysis(df, output_path, create_plots)
        
        # 4. Data Types and Distributions
        analysis_results['distributions'] = self._analyze_distributions(df, output_path, create_plots)
        
        # 5. Correlation Analysis
        analysis_results['correlations'] = self._correlation_analysis(df, output_path, create_plots)
        
        # 6. Outlier Detection
        analysis_results['outliers'] = self._outlier_detection(df, output_path, create_plots)
        
        # Save comprehensive report
        self._save_report(analysis_results, output_path)
        
        summary = self._create_summary(analysis_results)
        print(f"\n✓ EDA completed. Results saved to {output_dir}")
        
        return summary
    
    def _basic_info(self, df: pd.DataFrame) -> dict:
        """Extract basic dataset information"""
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
    
    def _summary_statistics(self, df: pd.DataFrame, output_path: Path) -> dict:
        """Generate and save summary statistics"""
        # Numeric columns
        numeric_stats = df.describe().to_dict()
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict()
            }
        
        # Save to CSV
        df.describe().to_csv(output_path / 'summary_statistics.csv')
        
        return {
            'numeric': numeric_stats,
            'categorical': categorical_stats
        }
    
    def _missing_data_analysis(self, df: pd.DataFrame, output_path: Path, 
                               create_plots: bool) -> dict:
        """Analyze missing data patterns"""
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_pct
        }).sort_values('missing_count', ascending=False)
        
        missing_df.to_csv(output_path / 'missing_data.csv')
        
        if create_plots and missing_counts.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df[missing_df['missing_count'] > 0]['missing_percentage'].plot(
                kind='barh', ax=ax
            )
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Data Analysis')
            plt.tight_layout()
            plt.savefig(output_path / 'missing_data_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return missing_df.to_dict()
    
    def _analyze_distributions(self, df: pd.DataFrame, output_path: Path, 
                               create_plots: bool) -> dict:
        """Analyze distributions of numeric and categorical variables"""
        results = {}
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if create_plots and len(numeric_cols) > 0:
            # Histograms
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                df[col].hist(ax=axes[idx], bins=30, edgecolor='black')
                axes[idx].set_title(col)
                axes[idx].set_ylabel('Frequency')
            
            # Hide empty subplots
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / 'distributions_histograms.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame, output_path: Path, 
                              create_plots: bool) -> dict:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {}
        
        corr_matrix = numeric_df.corr()
        corr_matrix.to_csv(output_path / 'correlation_matrix.csv')
        
        if create_plots:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True)
            ax.set_title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return corr_matrix.to_dict()
    
    def _outlier_detection(self, df: pd.DataFrame, output_path: Path, 
                           create_plots: bool) -> dict:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'n_outliers': len(outliers),
                'percentage': round(len(outliers) / len(df) * 100, 2)
            }
        
        if create_plots and len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            df[numeric_cols].boxplot(ax=ax, rot=45)
            ax.set_title('Boxplots for Outlier Detection')
            plt.tight_layout()
            plt.savefig(output_path / 'outliers_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return outlier_info
    
    def _save_report(self, results: dict, output_path: Path):
        """Save comprehensive JSON report"""
        with open(output_path / 'eda_report.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _create_summary(self, results: dict) -> str:
        """Create human-readable summary"""
        summary = f"""
Exploratory Data Analysis Complete
{'='*50}

Dataset Overview:
- Rows: {results['basic_info']['n_rows']:,}
- Columns: {results['basic_info']['n_columns']}
- Memory: {results['basic_info']['memory_usage']:.2f} MB

Missing Data:
- Total missing values detected
- Analysis saved to missing_data.csv

Visualizations Generated:
- Distribution histograms
- Correlation heatmap
- Outlier boxplots

All results saved to output directory.
        """
        return summary.strip()