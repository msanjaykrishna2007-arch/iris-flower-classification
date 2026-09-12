import pandas as pd
from pathlib import Path

def correct_historical_errors(df):
    """Corrects known discrepancies in the 35th and 38th samples of the UCI Iris dataset."""
    df_clean = df.copy()
    
    if len(df_clean) > 34 and df_clean.loc[34, 'species'] == 'Iris-setosa':
        df_clean.loc[34, 'petal_width'] = 0.2 
        
    if len(df_clean) > 37 and df_clean.loc[37, 'species'] == 'Iris-setosa':
        df_clean.loc[37, 'sepal_width'] = 3.6
        df_clean.loc[37, 'petal_length'] = 1.4
        
    return df_clean

def clean_data(df):
    """Checks for missing values, corrects anomalies, and drops duplicates."""
    df = correct_historical_errors(df)
    df = df.dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def separate_features_target(df, target_column='species'):
    """Splits the dataframe into predictive features (X) and target class (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_features(X):
    """Applies standard scaling to the numeric features manually (Z-score normalization)."""
    X_scaled = (X - X.mean()) / X.std(ddof=0)
    return X_scaled

def export_clean_data(df, output_path):
    """Saves the cleaned dataset to a new CSV file for other team members to use."""
    # Ensure the parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully exported to: {output_path}")

def run_preprocessing_pipeline(df, target_column='species', export_path=None):
    """Wrapper function to execute the full preprocessing workflow."""
    # 1. Clean the data
    df_clean = clean_data(df)
    
    # 2. Export the unscaled, clean data if a path is provided
    if export_path:
        export_clean_data(df_clean, export_path)
        
    # 3. Separate and scale for the Machine Learning members
    X, y = separate_features_target(df_clean, target_column)
    X_scaled = scale_features(X)
    
    return X_scaled, y