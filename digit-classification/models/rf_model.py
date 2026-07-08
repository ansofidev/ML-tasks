from models.interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier


# Random Forest classifier
class RFClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=0):
        # Initialize Random Forest model with given parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,  # Number of trees in the forest
            random_state=random_state   # Seed for reproducibility
        )


    # Train the Random Forest model
    def train(self, X_train, y_train):
        print("Training Random Forest...")
        # Fit the model on training data
        self.model.fit(X_train, y_train) 
        print("Random Forest training finished")


    # Predictions for new samples
    def predict(self, X):
        return self.model.predict(X)