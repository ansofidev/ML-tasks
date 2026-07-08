import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.interface import MnistClassifierInterface


# Feed-forward Neural Network classifier
class NNClassifier(MnistClassifierInterface):
    # MNIST images are 28x28 pixels which equel to 784 input features
    def __init__(self, input_size=784, num_classes=10):
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  # First hidden layer with 128 neurons
            nn.ReLU(),                   # ReLU activation: max(0, x)
            nn.Linear(128, 64),          # Second hidden layer with 64 neurons (feature compression)
            nn.ReLU(),
            nn.Linear(64, num_classes)   # Output scores for digits 0–9
        )

        # Loss function that measures how different predictions are from true labels
        self.loss_fn = nn.CrossEntropyLoss()
        # Adam optimizer with learning rate 0.001 to adjust model weights during training
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    """"
    Train the neural network
    epochs is number of times the model will iterate over the entire training dataset
    batch_size is number of samples processed before updating model weights
    """
    def train(self, X_train, y_train, epochs=50, batch_size=64):
        # Convert training data to PyTorch tensors
        X = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)

        # Assign inputs to labels
        dataset = TensorDataset(X, y)
        # Enables mini-batch training and shuffles data each epoch to increase generalization
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0

            # Add progress bar
            for batch_X, batch_y in tqdm(loader, desc=f"NN Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()      # Reset gradients from the previous batch so they don't accumulate
                outputs = self.model(batch_X)   # Forward pass: get predictions for the batch
                loss = self.loss_fn(outputs, batch_y)  # Сompute error
                loss.backward()                 # Compute gradients showing how each weight affects the error
                self.optimizer.step()           # Update the model weights

                epoch_loss += loss.item()

            # Loss during the epoch
            print(f"loss: {epoch_loss:.4f}")


    # Predictions for new samples
    def predict(self, X):
        # Convert data to tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Disable gradient calculation because we only need inference (evaluation)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.numpy()