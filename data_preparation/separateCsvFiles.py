import pandas as pd
import os

# Load the main CSV file
df = pd.read_csv('ds_train-n1-1-csv_files/ds_train_n1-1_filtered.csv')

# Unique combinations of 'machine' and 'app'
combinations = df[['machine', 'app']].drop_duplicates()

# Ensure the directory for the smaller csv files exists
output_dir = 'ds_train-n1-1-csv_files'
os.makedirs(output_dir, exist_ok=True)

for _, row in combinations.iterrows():
    # Filter rows for the current combination
    df_subset = df[(df['machine'] == row['machine']) & (df['app'] == row['app'])]
    # File name based on 'machine' and 'app' values
    file_name = f"{row['machine']}_{row['app']}.csv"
    file_path = os.path.join(output_dir, file_name)
    # Save to a new CSV file
    df_subset.to_csv(file_path, index=False)

print("CSV files created successfully.")

