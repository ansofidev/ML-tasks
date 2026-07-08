import numpy as np
import matplotlib.pyplot as plt


def show_misclassified_examples(X_test, y_test, predictions, n=5):
    misclassified = np.where(predictions != y_test)[0]

    # Randomly choose some errors
    idx = np.random.choice(misclassified, n)

    _, axes = plt.subplots(1, n, figsize=(10,3))

    for i, ax in enumerate(axes):
        sample = idx[i]

        ax.imshow(X_test[sample].reshape(28,28), cmap="gray") # Reshape flat vector back to 28x28 MNIST format
        ax.set_title(f"Pred:{predictions[sample]}\nTrue:{y_test[sample]}") # Show predicted vs actual label
        ax.axis("off")

    plt.show()