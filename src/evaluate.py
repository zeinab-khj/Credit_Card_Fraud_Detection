from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):

    y_proba = model.predict_proba(X_test)[:, 1]
    

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])

    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_idx = f1_scores.argmax()

    best_threshold = thresholds[best_idx]

    y_pred = (y_proba >= best_threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred
        ),
    }

    return results
