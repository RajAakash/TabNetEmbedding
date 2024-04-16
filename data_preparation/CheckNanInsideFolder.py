import pandas as pd
import os

def check_nan_in_files(directory):
    # List to keep track of files with NaN values
    files_with_nan = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):  # check if the file is a CSV
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)

            # Check if there are any NaN values in the DataFrame
            if data.isnull().values.any():
                files_with_nan.append(filename)

    return files_with_nan

# Specify the directory to check
directory_path = r'C:\Users\ACER\Downloads\Newfolder\q-c\src'
files_with_nan = check_nan_in_files(directory_path)

if files_with_nan:
    print("Files with NaN values:", files_with_nan)
else:
    print("No files with NaN values found.")            #feature_30_df1