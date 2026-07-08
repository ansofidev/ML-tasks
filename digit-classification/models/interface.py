from abc import ABC, abstractmethod

# Abstract interface for all MNIST classifiers
class MnistClassifierInterface(ABC):

    # Model training method
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    # Method for making predictions
    @abstractmethod
    def predict(self, X):
        pass