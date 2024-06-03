import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from typing import Tuple
mlflow.set_tracking_uri("http://192.168.1.159:5000")
mlflow.set_registry_uri("http://192.168.1.159:5000")

experiment_name  = "pytorch101"
mlflow.set_experiment(experiment_name)
# Define constants
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = './data'
MODEL_DIR = './model'

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def prepare_data(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """Prepare the FashionMNIST dataset.

    Args:
        data_dir (str): Directory where the dataset will be stored.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Load the FashionMNIST dataset
    dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

class FashionMNISTModel(nn.Module):
    """Simple neural network for FashionMNIST classification."""
    
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam) -> float:
    """Train the model.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.CrossEntropyLoss): Loss function.
        optimizer (optim.Adam): Optimizer.

    Returns:
        float: Training loss.
    """
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def evaluate_model(model: nn.Module, val_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> Tuple[float, float]:
    """Evaluate the model.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.CrossEntropyLoss): Loss function.

    Returns:
        Tuple[float, float]: Validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    
    return val_loss, accuracy

def main():
    """Main function to run the training and evaluation pipeline."""
    
    # Prepare data
    train_loader, val_loader = prepare_data(DATA_DIR)
    
    # Initialize model, criterion, optimizer
    model = FashionMNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # MLflow tracking
    with mlflow.start_run(run_name="Exploratory run"):
        mlflow.log_param('epochs', EPOCHS)
        mlflow.log_param('batch_size', BATCH_SIZE)
        mlflow.log_param('learning_rate', LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{EPOCHS}.. Train loss: {train_loss:.3f}.. '
                  f'Val loss: {val_loss:.3f}.. Val accuracy: {val_accuracy:.3f}')
            
            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_accuracy, step=epoch)
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'fashion_mnist_model.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, 'model')
        
        # Log artifacts
        mlflow.log_artifact(DATA_DIR)
        mlflow.log_artifact(MODEL_DIR)

if __name__ == '__main__':
    main()
