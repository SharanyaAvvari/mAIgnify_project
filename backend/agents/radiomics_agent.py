from smolagents import MultiStepAgent
from tools.radiomic_tool import PyRadiomicsFeatureExtractionTool
from typing import List


class RadiomicAgent:
    """Agent for radiomic feature extraction from medical images"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.tool = PyRadiomicsFeatureExtractionTool()
        
        self.agent = MultiStepAgent(
            tools=[self.tool],
            model=self.llm,
            max_steps=10,
            verbose=True
        )
    
    def get_tools(self) -> List:
        return [self.tool]
    
    def is_ready(self) -> bool:
        return self.tool is not None