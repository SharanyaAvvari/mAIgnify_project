from smolagents import MultiStepAgent
from tools.nnunet_training_tool import NNUNetTrainingTool
from tools.nnunet_inference_tool import NNUNetInferenceTool
from typing import List


class NNUNetAgent:
    """Agent for nnU-Net segmentation model training and inference"""
    
    def __init__(self, llm_model):
        self.llm = llm_model
        self.training_tool = NNUNetTrainingTool()
        self.inference_tool = NNUNetInferenceTool()
        
        self.agent = MultiStepAgent(
            tools=[self.training_tool, self.inference_tool],
            model=self.llm,
            max_steps=15,
            verbose=True
        )
    
    def get_tools(self) -> List:
        return [self.training_tool, self.inference_tool]
    
    def is_ready(self) -> bool:
        return self.training_tool is not None and self.inference_tool is not None

