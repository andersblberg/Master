import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def train_nn_model(X, y):
    """
    Trains a feed-forward Neural Network using MLPClassifier on the given dataset.

    Hyperparameters:
      - hidden_layer_sizes=(64, 32): Two hidden layers with 64 and 32 neurons, respectively.
      - activation='relu': ReLU activation function.
      - solver='adam': Adam optimizer.
      - max_iter=500: Maximum number of iterations for convergence.
      - random_state=42: Ensures reproducible results.

      dict: A dictionary with keys:
            - 'model': the trained MLPClassifier,
            - 'accuracy': the accuracy on the held-out test set,
            - 'params': model hyperparameters (from get_params()).
    """
    # Split data into training and test sets --- Legacy
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Initialize the MLPClassifier with the desired hyperparameters.
    model = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {"model": model, "accuracy": acc, "params": model.get_params()}


if __name__ == "__main__":
    import numpy as np

    X_dummy = np.random.rand(100, 8)
    y_dummy = np.random.choice(["Class1", "Class2"], size=100)

    results = train_nn_model(X_dummy, y_dummy)
    print("Dummy NN Model Accuracy:", results["accuracy"])
