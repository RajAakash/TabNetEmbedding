import pandas as pd

# Load the CSV file
file_path = 'C:/Users/ACER/Downloads/concatenation_with_source/q-r/tar/newkripke-qr.csv'
data = pd.read_csv(file_path)

# Check each column for NaN values and report
for column in data.columns:
    nan_count = data[column].isna().sum()
    if nan_count > 0:
        print(f"The column '{column}' contains {nan_count} NaN values.")
    else:
        print(f"The column '{column}' does not contain any NaN values.")