"""
Exploratory Data Analysis (EDA) Agent
Performs comprehensive EDA on tabular datasets
"""

from smolagents import MultiStepAgent
from tools.eda_tool import ExploratoryDataAnalysisTool
from typing import List


class EDAAgent:
    """Agent responsible for exploratory data analysis"""
    
    def __init__(self, llm_model):
        """
        Initialize EDA Agent
        
        Args:
            llm_model: Language model for reasoning
        """
        self.llm = llm_model
        self.tool = ExploratoryDataAnalysisTool()
        
        # Create agent with EDA tool
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
        """Check if agent is ready to execute tasks"""
        return self.tool is not None
    
    def execute(self, query: str) -> dict:
        """
        Execute EDA task based on natural language query
        
        Args:
            query: Natural language instruction for EDA
            
        Returns:
            Dictionary with execution results
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