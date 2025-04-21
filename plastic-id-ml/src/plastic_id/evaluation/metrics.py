from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'confusion': confusion_matrix(y_true, y_pred).tolist(),
    }