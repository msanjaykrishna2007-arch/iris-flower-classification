from pathlib import Path
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Add the project root to Python's import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.model import run_classification


def evaluate_model(y_test, y_pred):
    """
    Calculate standard classification evaluation metrics.
    """
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix
    }


def run_evaluation(file_path):
    """
    Run the classification workflow and evaluate its predictions.
    """
    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred
    ) = run_classification(file_path)

    results = evaluate_model(y_test, y_pred)

    return model, X_test, y_test, y_pred, results


if __name__ == "__main__":
    data_path = project_root / "data" / "iris_cleaned.csv"

    (
        model,
        X_test,
        y_test,
        y_pred,
        results
    ) = run_evaluation(data_path)

    print("Iris model evaluation completed successfully.")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1-score: {results['f1_score']:.2%}")

    print("\nClassification Report:")
    print(results["classification_report"])

    print("Confusion Matrix:")
    print(results["confusion_matrix"])
