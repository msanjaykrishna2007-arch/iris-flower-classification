import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_cleaned_data(file_path):
    """
    Load the cleaned Iris dataset prepared by the preprocessing stage.
    """
    return pd.read_csv(file_path)


def prepare_data(df, target_column="species"):
    """
    Separate input features (X) from the target variable (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split the dataset into training and testing sets.

    Stratification keeps the class distribution balanced between
    the training and testing datasets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_logistic_regression_model():
    """
    Create the Logistic Regression classification pipeline.

    StandardScaler is fitted only on the training data through
    the Pipeline, which helps prevent data leakage.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    return model


def train_model(model, X_train, y_train):
    """
    Train the classification model using the training data.
    """
    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    """
    Generate predictions for unseen test data.
    """
    return model.predict(X_test)


def run_classification(file_path):
    """
    Run the complete Iris classification workflow.

    Returns the trained model, test features, test labels,
    and predictions so that the evaluation stage can use them.
    """
    df = load_cleaned_data(file_path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = create_logistic_regression_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    y_pred = predict(
        model,
        X_test
    )

    return model, X_train, X_test, y_train, y_test, y_pred


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "iris_cleaned.csv"

    (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred
    ) = run_classification(data_path)

    print("Iris classification completed successfully.")
    print(f"Total samples: {len(y_train) + len(y_test)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Predictions generated: {len(y_pred)}")
