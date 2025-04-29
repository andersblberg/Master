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
      
    The function splits the data into training (80%) and testing (20%) sets using stratified sampling 
    based on y, then trains the model, and finally returns the trained model, test-set accuracy, and the model parameters.
    
    Parameters:
      X (array-like): Feature matrix.
      y (array-like): Target labels.
      
    Returns:
      dict: A dictionary with keys:
            - 'model': the trained MLPClassifier,
            - 'accuracy': the accuracy on the held-out test set,
            - 'params': model hyperparameters (from get_params()).
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # ensures that both splits have the same class proportions
    )
    
    # Initialize the MLPClassifier with the desired hyperparameters.
    model = MLPClassifier(
        hidden_layer_sizes=(256, 256, 256),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    # Train the model on the training data.
    model.fit(X_train, y_train)
    
    # Predict on the test set.
    y_pred = model.predict(X_test)
    
    # Calculate accuracy.
    acc = accuracy_score(y_test, y_pred)
    
    # Return the results in a dictionary.
    return {
        "model": model,
        "accuracy": acc,
        "params": model.get_params()
    }

if __name__ == "__main__":
    # For standalone testing: generate dummy data.
    import numpy as np
    # Create a dummy feature matrix with 100 samples and 8 features
    X_dummy = np.random.rand(100, 8)
    # Create dummy targets with two classes.
    y_dummy = np.random.choice(["Class1", "Class2"], size=100)
    
    results = train_nn_model(X_dummy, y_dummy)
    print("Dummy NN Model Accuracy:", results["accuracy"])
