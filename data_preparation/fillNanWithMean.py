import os
import pandas as pd

# Set the directory path to the folder containing your CSV files
directory_path = r'C:\Users\ACER\Downloads\Newfolder\q-c\tar'

# Loop through every file in the specified directory
for filename in os.listdir(directory_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)  # Full path to the file
        
        # Load the data from the CSV file
        df = pd.read_csv(file_path)
        
        # Iterate over each column in the DataFrame
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:  # Check if the column is numeric
                mean_value = df[column].mean()  # Calculate the mean excluding NaN values
                df[column].fillna(value=mean_value, inplace=True)  # Fill NaN values with the mean

        # Save the modified DataFrame back to the same CSV file
        df.to_csv(file_path, index=False)
        print(f"Updated file: {filename}")

print("All CSV files in the directory have been updated.")
