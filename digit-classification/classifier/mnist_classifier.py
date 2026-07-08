from dataclasses import dataclass
from typing import Type

from models.rf_model import RFClassifier
from models.nn_model import NNClassifier
from models.cnn_model import CNNClassifier
from models.interface import MnistClassifierInterface


@dataclass
class AlgorithmType:
    model: Type  # Class of the model 
    name: str    # Name of the algorithm


# Dictionary mapping algorithm keys to model classes and their names
MODELS = {
    "rf": AlgorithmType(RFClassifier, "Random Forest"),
    "nn": AlgorithmType(NNClassifier, "Feed-Forward Neural Network"),
    "cnn": AlgorithmType(CNNClassifier, "Convolutional Neural Network"),
}


class MnistClassifier(MnistClassifierInterface):
    # Wrapper class that selects and uses the chosen MNIST classification algorithm
    def __init__(self, algorithm: str):

        # Check if the selected algorithm exists in the registry
        if algorithm in MODELS:
            # Instantiate the selected model
            self.model = MODELS[algorithm].model()  
        else:
            # Raise error if algorithm key is invalid
            raise ValueError("Invalid input")  

    # Train the model
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    # Make predictions
    def predict(self, X_test):
        return self.model.predict(X_test)