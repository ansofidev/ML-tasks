from data.mnist_loader import load_mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from classifier import mnist_classifier


def main():
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = load_mnist()

    # Iterate over all defined models
    for key, algo in mnist_classifier.MODELS.items():
        # Create classifier based on the selected algorithm
        model = mnist_classifier.MnistClassifier(key)

        # Train the model
        model.train(X_train, y_train)
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Print results
        print("-" * 100)
        print(f"Model: {algo.name}")
        print()
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print()


# Run this file directly
if __name__ == "__main__":
    main()