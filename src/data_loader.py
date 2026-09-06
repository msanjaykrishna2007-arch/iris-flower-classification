import pandas as pd


COLUMN_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]


def load_iris_data(file_path="data/iris.data"):
    """
    Load the official UCI Iris dataset.
    """
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMN_NAMES
    )

    # Remove completely empty rows, if present.
    df = df.dropna(how="all")

    return df


if __name__ == "__main__":
    iris_df = load_iris_data()

    print("Iris dataset loaded successfully.")
    print(f"Rows: {iris_df.shape[0]}")
    print(f"Columns: {iris_df.shape[1]}")

    print("\nColumn names:")
    print(iris_df.columns.tolist())

    print("\nFirst 5 rows:")
    print(iris_df.head())