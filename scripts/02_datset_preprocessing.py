import sys
from pathlib import Path
import pandas as pd

try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_iris_data
import src.preprocessing as pp

def main():
    print("=" * 60)
    print("IRIS FLOWER CLASSIFICATION - PREPROCESSING PIPELINE")
    print("=" * 60)

    # Define input and output paths
    input_path = PROJECT_ROOT / "data" / "iris.data"
    output_path = PROJECT_ROOT / "data" / "iris_cleaned.csv"
    
    try:
        df_raw = load_iris_data(input_path)
        print(f"\nRaw data loaded successfully. Initial Shape: {df_raw.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\n--- Running Preprocessing ---")
    
    # Run pipeline and export the clean data
    X_scaled, y = pp.run_preprocessing_pipeline(
        df_raw, 
        target_column='species', 
        export_path=output_path
    )
    
    print("\n--- Preprocessing Complete ---")
    print(f"Final Features Shape (Duplicates Removed): {X_scaled.shape}")
    print(f"Final Target Shape: {y.shape}")

if __name__ == "__main__":
    main()