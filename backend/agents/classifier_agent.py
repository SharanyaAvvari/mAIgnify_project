"""
Classifier Agent
Trains and deploys classification models on tabular data
"""

from smolagents import MultiStepAgent
from tools.pycaret_classification_tool import PyCaretClassificationTool
from tools.pycaret_inference_tool import PyCaretInferenceTool
from typing import List


class ClassifierAgent:
    """Agent for classification model development and inference"""
    
    def __init__(self, llm_model):
        """
        Initialize Classifier Agent
        
        Args:
            llm_model: Language model for reasoning
        """
        self.llm = llm_model
        self.training_tool = PyCaretClassificationTool()
        self.inference_tool = PyCaretInferenceTool()
        
        self.agent = MultiStepAgent(
            tools=[self.training_tool, self.inference_tool],
            model=self.llm,
            max_steps=15,
            verbose=True
        )
    
    def get_tools(self) -> List:
        """Return list of tools"""
        return [self.training_tool, self.inference_tool]
    
    def is_ready(self) -> bool:
        """Check if agent is ready"""
        return self.training_tool is not None and self.inference_tool is not None
    
    def execute(self, query: str) -> dict:
        """Execute classification task"""
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