from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np


# Load MNIST dataset and split it into training and testing sets
def load_mnist(test_size=0.2, random_state=0):
    # Download MNIST dataset from OpenML
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    # Normalize pixel values to [0, 1] (MNIST images use 8-bit grayscale values 0–255)
    X = mnist.data.astype(np.float32) / 255.0
    # Convert labels to integer type
    y = mnist.target.astype(np.int64)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution in both splits
    )

    return X_train, X_test, y_train, y_test