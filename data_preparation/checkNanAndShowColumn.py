import pandas as pd

# Load the CSV file
file_path = 'C:/Users/ACER/Downloads/concatenation_with_source/q-r/tar/newkripke-qr.csv'  # Update this to your CSV file path
data = pd.read_csv(file_path)

# Initialize a flag to indicate if any NaNs are found
nan_found = False

# Check each column for NaN values
for column in data.columns:
    if data[column].isna().any():  # Check if there are any NaNs in the column
        nan_found = True
        nan_count = data[column].isna().sum()
        print(f"Column '{column}' contains {nan_count} NaN values.")
        
if not nan_found:
    print("No NaN values found in any column.")

