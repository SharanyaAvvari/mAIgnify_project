"""
Image Classifier Agent
Orchestrates image classification tasks using PyTorch tools.
"""

from smolagents import Agent
# Note: Adjust the import paths based on your final project structure.
from tools.pytorch_resnet_training_tool import PyTorchResNetTrainingTool
from tools.pytorch_vgg16_training_tool import PyTorchVGG16TrainingTool
from tools.pytorch_inceptionv3_training_tool import PyTorchInceptionV3TrainingTool
from tools.pytorch_resnet_inference_tool import PyTorchResNetInferenceTool
# To be complete, this agent should also have inference tools for VGG and Inception.
# You would create them by copying the ResNet inference tool and modifying the _get_model method.

class ImageClassifierAgent(Agent):
    """
    An agent that specializes in end-to-end image classification.
    It can train various architectures (ResNet, VGG16, InceptionV3)
    and perform inference using the trained models.
    """
    
    def __init__(self):
        super().__init__(
            name="ImageClassifierAgent",
            description="""
            This agent performs automated image classification.
            Given a dataset of images, it can train different deep learning models
            (like ResNet, VGG16, InceptionV3), save the best performing model,
            and then use that model to predict classes for new, unseen images.
            """,
            tools=[
                PyTorchResNetTrainingTool(),
                PyTorchVGG16TrainingTool(),
                PyTorchInceptionV3TrainingTool(),
                PyTorchResNetInferenceTool(),
                # Add PyTorchVGG16InferenceTool() and PyTorchInceptionV3InferenceTool() here
                # once you create them by adapting the ResNet inference tool.
            ]
        )