import pandas as pd

# Load the CSV file
file_path = 'C:/Users/ACER/Downloads/concatenation_with_source/q-r/tar/kripke-qr.csv'  # Update this to your CSV file path
data = pd.read_csv(file_path)

# Replace NaNs with the mean of the column
for column in data.columns:
    if data[column].isna().any():  # Check if there are any NaNs in the column
        mean_value = data[column].mean()  # Calculate mean
        data[column].fillna(mean_value, inplace=True)  # Replace NaNs with the mean

# Save the modified DataFrame back to CSV
data.to_csv('C:/Users/ACER/Downloads/concatenation_with_source/q-r/tar/newkripke-qr.csv', index=False)

print("NaNs have been replaced with the mean of their respective columns.")
