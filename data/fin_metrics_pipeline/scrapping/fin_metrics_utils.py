import pandas as pd


def read_csv_file(csv_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None


def save_to_csv(df, csv_path):
    """Saves a DataFrame to a specified CSV file."""
    try:
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    except Exception as e:
        print(f"Error saving data to {csv_path}: {e}")
