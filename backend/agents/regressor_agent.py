"""
Regressor Agent
Orchestrates regression tasks using PyCaret tools.
"""

from smolagents import Agent
# Note: Adjust the import path based on your project structure.
# Assuming your tools are in a 'tools' directory.
from tools.pycaret_regression_tool import PyCaretRegressionTool
from tools.pycaret_regression_inference_tool import PyCaretRegressionInferenceTool

class RegressorAgent(Agent):
    """
    An agent specialized for handling end-to-end regression tasks,
    including model training and inference.
    """
    
    def __init__(self):
        super().__init__(
            name="RegressorAgent",
            description="""
            This agent performs automated regression. 
            Given a dataset and a target column, it can train multiple models,
            tune the best ones, create a blended ensemble, and save the results.
            It can also use a trained model to make predictions on new data.
            """,
            tools=[
                PyCaretRegressionTool(),
                PyCaretRegressionInferenceTool()
            ]
        )