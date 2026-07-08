import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.interface import MnistClassifierInterface

# Convolutional Neural Network classifier
class CNNClassifier(MnistClassifierInterface):
    def __init__(self, image_size=28, num_classes=10):
        self.image_size = image_size
        # # Feature size after two MaxPool(2) layers
        feature_size = image_size // 4

        """"
        Define CNN architecture
        kernel_size is the size of the filter that scans the image
        padding adds a 1-pixel border so the output size stays the same
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # First conv layer, 1 channel because MNIST is grayscale
            nn.ReLU(),  # ReLU activation: max(0, x) 
            nn.MaxPool2d(2),  # 2x2 pooling window to reduce image size (helps reduce parameters and overfitting)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),  # Convert 2D feature maps to 1D vector for linear layers

            nn.Linear(64 * feature_size * feature_size, 128), # Combine extracted features into 128 neurons
            nn.ReLU(), #Add non-linearity to learn more complex patterns
            nn.Linear(128, num_classes)  # Output scores for digits 0–9
        )

        # Loss function that measures how different predictions are from true labels
        self.loss_fn = nn.CrossEntropyLoss()
        # Adam optimizer with learning rate 0.001 to adjust model weights during training
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


    def train(self, X_train, y_train, epochs=10, batch_size=64):
        # Convert training data to PyTorch tensors
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)

        # CNN expects images with shape (batch, channels, height, width)
        X = X.view(-1, 1, self.image_size, self.image_size)

        # Assign inputs to labels
        dataset = TensorDataset(X, y)
        # Enables mini-batch training and shuffles data each epoch to increase generalization
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Enable evaluation mode to disable training mechanisms
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0

            # Add progress bar
            for batch_X, batch_y in tqdm(loader, desc=f"CNN Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()  # Reset gradients from the previous batch so they don't accumulate
                outputs = self.model(batch_X)  # Forward pass: get predictions for the batch
                loss = self.loss_fn(outputs, batch_y)  # Сompute error
                loss.backward()  # Compute gradients showing how each weight affects the error
                self.optimizer.step()  # Update weights
                epoch_loss += loss.item()

            # Loss during the epoch
            print(f"loss: {epoch_loss:.4f}")


    # Predictions for new samples
    def predict(self, X):
        # Convert input data to tensor
        X = torch.tensor(X, dtype=torch.float32)
        # Reshape because CNN expects 4D input
        X = X.view(-1, 1, self.image_size, self.image_size)

        # Enable evaluation mode to disable training mechanisms
        self.model.eval() 

        # Disable gradient calculation because we only need inference (evaluation)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()