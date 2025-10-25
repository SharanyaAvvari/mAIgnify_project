from smolagents import MultiStepAgent
from tools.totalsegmentator_tool import TotalSegmentatorTool
from typing import List


class TotalSegmentatorAgent:
    """Agent for anatomical structure segmentation using TotalSegmentator"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.tool = TotalSegmentatorTool()
        
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