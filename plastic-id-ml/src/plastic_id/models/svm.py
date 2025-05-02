# SVM/SVM_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm_model(X, y):
    """
    Trains an SVM and returns a dict with {'model', 'accuracy'}.
    """
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": acc
    }
