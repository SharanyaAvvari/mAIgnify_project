from smolagents import MultiStepAgent
from tools.tumor_classification_tool import TumorClassificationTool

class TumorAgent:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.tool = TumorClassificationTool()
        
        self.agent = MultiStepAgent(
            tools=[self.tool],
            model=self.llm,
            max_steps=10,
            verbose=True
        )
    
    def get_tools(self):
        return [self.tool]