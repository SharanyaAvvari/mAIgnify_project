"""
Feature Importance Analysis Agent
Identifies most relevant features for ML tasks
"""

from smolagents import MultiStepAgent
from tools.feature_importance_tool import FeatureImportanceAnalysisTool
from typing import List


class FeatureImportanceAgent:
    """Agent responsible for feature importance analysis and selection"""
    
    def __init__(self, llm_model):
        """
        Initialize Feature Importance Agent
        
        Args:
            llm_model: Language model for reasoning
        """
        self.llm = llm_model
        self.tool = FeatureImportanceAnalysisTool()
        
        self.agent = MultiStepAgent(
            tools=[self.tool],
            model=self.llm,
            max_steps=10,
            verbose=True
        )
    
    def get_tools(self) -> List:
        """Return list of tools used by this agent"""
        return [self.tool]
    
    def is_ready(self) -> bool:
        """Check if agent is ready"""
        return self.tool is not None
    
    def execute(self, query: str) -> dict:
        """
        Execute feature importance analysis
        
        Args:
            query: Natural language instruction
            
        Returns:
            Dictionary with results
        """
        try:
            result = self.agent.run(query)
            return {
                'status': 'success',
                'result': result
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }