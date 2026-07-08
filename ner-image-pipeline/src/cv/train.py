import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utils.device import get_device
from tqdm import tqdm


# Define model
def build_model(num_classes=10):
    # Create ResNet18 architecture without pretrained weights
    model = models.resnet18(weights=None)
    # Number of features from backbone before classification layer
    in_features = model.fc.in_features

    # Replace final layer with dropout + linear classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Helps prevent overfitting by deactivating random 50% of neurons
        nn.Linear(in_features, num_classes)  # Output equals number of animal classes
    )

    return model


def get_transforms():
    # Data augmentation for training to improve predictions
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),  # Resize first to allow cropping
        transforms.RandomResizedCrop(224),  # Standard size for ResNet
        transforms.RandomHorizontalFlip(),  # Some examples will be flipped
        transforms.RandomRotation(15),  # Some examples will be rotated by 15 degrees
        transforms.ColorJitter( # Apply color augmentation
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ), 
        transforms.ToTensor(),
        transforms.Normalize( # ImageNet normalization 
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )  
    ])

    # Validation transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),  # Direct resize for evaluation because ResNet expects this size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return train_transform, val_transform


def build_dataloaders(data_dir, batch_size=32):
    train_transform, val_transform = get_transforms()

    # Load dataset from folder structure
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)

   # Define sizes for splitting
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    # Split data with shufling
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply validation transforms to validation dataset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 32 for good balance between speed and memory
        shuffle=True,  # Shuffle training data for better learning
        num_workers=2  # Parallel loading of images
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=2
    )

    return train_loader, val_loader, dataset.classes


def evaluate(model, loader, device):
    # Enable evaluation mode to disable training mechanisms
    model.eval()

    correct = 0
    total = 0

    # Disable gradient calculation because we only need inference (evaluation)
    with torch.no_grad():
        for images, labels in loader:

            # Move images to the same device as model
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images) # Model make predictions
            preds = torch.argmax(outputs, dim=1) # Choose the most probable class

            correct += (preds == labels).sum().item() # Count correct predictions
            total += labels.size(0)

    # Return accuracy
    return correct / total


# Model training
def train_model(data_dir, save_path="models/cv_model/resnet18.pth", epochs=20):
    device = get_device()

    train_loader, val_loader, classes = build_dataloaders(data_dir)

    model = build_model(num_classes=len(classes))
    model.to(device)

    # Cross entropy for multi-class classification, label_smoothing helps reduce overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Adam optimizer with learning rate 0.001 to adjust model weights during training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4  # L2 regularization to prevent overfitting
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,  # # Reduce learning rate every 5 epochs to help the model reach the minimum
        gamma=0.5  # Multiply Learning rate by 0.5
    )

    for epoch in range(epochs):
        model.train()

        correct = 0
        total = 0
        train_loss = 0

        # Add progress bar
        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Reset gradients from the previous batch so they don't accumulate
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # Compute gradients showing how each weight affects the error
            optimizer.step() # Update weights
            train_loss += loss.item() * images.size(0) # Calculate loss

            preds = torch.argmax(outputs, dim=1) 

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total  # Calculate train accuracy
        train_loss /= total

        scheduler.step()  # Update learning rate

        val_acc = evaluate(model, val_loader, device) # Calculate validation accuracy

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    # Save trained model
    os.makedirs("models/cv_model", exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print("Model saved to:", save_path)

    return classes