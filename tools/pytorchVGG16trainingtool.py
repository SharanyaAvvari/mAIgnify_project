"""
PyTorch VGG16 Training Tool
"""

from smolagents import Tool
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from pathlib import Path
import json
import pandas as pd
import time
import copy

class PyTorchVGG16TrainingTool(Tool):
    """Tool for training VGG16 image classification models."""
    
    name = "pytorch_vgg16_training"
    description = """Trains a VGG16 model for image classification.
    
    Inputs:
    - train_dir: Directory for training images.
    - val_dir: Directory for validation images.
    - output_dir: Directory to save the model and results.
    - num_classes: The number of classes in the dataset.
    - num_epochs: Number of training epochs (default: 25).
    - batch_size: Training batch size (default: 32).
    - learning_rate: Optimizer learning rate (default: 0.001).
    - pretrained: Whether to use a pretrained model (default: True).
    
    Outputs:
    - The best trained model (.pt file).
    - Training history and performance metrics (JSON).
    - A summary report of the training process.
    """
    
    inputs = {
        "train_dir": {"type": "string", "description": "Path to the training data directory"},
        "val_dir": {"type": "string", "description": "Path to the validation data directory"},
        "output_dir": {"type": "string", "description": "Directory to save outputs"},
        "num_classes": {"type": "integer", "description": "Number of target classes"},
        "num_epochs": {"type": "integer", "description": "Number of epochs to train for", "default": 25},
        "batch_size": {"type": "integer", "description": "Batch size for training", "default": 32},
        "learning_rate": {"type": "number", "description": "Learning rate for the optimizer", "default": 0.001},
        "pretrained": {"type": "boolean", "description": "Use pretrained weights", "default": True}
    }
    
    output_type = "string"
    
    def forward(self, train_dir: str, val_dir: str, output_dir: str, num_classes: int,
                num_epochs: int = 25, batch_size: int = 32, 
                learning_rate: float = 0.001, pretrained: bool = True) -> str:
        
        # --- This training logic is identical to the ResNet tool ---
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }
        dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
            for x in ['train', 'val']
        }
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        # Get model (VGG16 specific)
        model = self._get_model(num_classes, pretrained)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        print(f"🚀 Starting VGG16 training for {num_epochs} epochs on {device}...")

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}'); print('-' * 10)
            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss, running_corrects = 0.0, 0
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward(); optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), output_path / 'best_model.pt')
                    print("✨ New best model saved!")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        
        with open(output_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        return self._create_summary(output_path, time_elapsed, best_acc)
    
    def _get_model(self, num_classes, pretrained):
        """Key Change: Loads VGG16 and replaces the final classifier layer."""
        model = models.vgg16(pretrained=pretrained)
        # Freeze all the network parameters
        for param in model.features.parameters():
            param.requires_grad = False
        # Replace the final layer
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        return model

    def _create_summary(self, output_path, time_elapsed, best_acc) -> str:
        return f"""
PyTorch VGG16 Training Complete
{'='*60}
Training Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s
Best Validation Accuracy: {best_acc:.4f}

Outputs saved to: {output_path}
- Best model: best_model.pt
- Training history: training_history.json
"""