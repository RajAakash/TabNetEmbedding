import csv
import os

# Paths to the two folders
folder1_path = 'ds_train-n1-1-csv_files/relative-quartz'
folder2_path = 'ds_train-n1-1-csv_files/relative-corona'

# New folder paths
new_folder1_path = os.path.join(folder2_path, 'folder1')
new_folder2_path = os.path.join(folder1_path, 'folder2')

# Create the new folders if they don't exist
os.makedirs(new_folder1_path, exist_ok=True)
os.makedirs(new_folder2_path, exist_ok=True)

# List all CSV files in the first folder
csv_files_folder1 = [f for f in os.listdir(folder1_path) if f.endswith('.csv')]

for csv_file in csv_files_folder1:
    # Construct file paths for both original and new folders
    file_path_folder1 = os.path.join(folder1_path, csv_file)
    file_path_folder2 = os.path.join(folder2_path, csv_file)
    new_file_path_folder1 = os.path.join(new_folder2_path, csv_file)
    new_file_path_folder2 = os.path.join(new_folder1_path, csv_file)

    # Check if the corresponding file exists in folder2
    if os.path.exists(file_path_folder2):
        # Read both files
        with open(file_path_folder1, mode='r', newline='', encoding='utf-8') as file1, \
             open(file_path_folder2, mode='r', newline='', encoding='utf-8') as file2:
            reader1 = list(csv.DictReader(file1))
            reader2 = list(csv.DictReader(file2))

            fieldnames1 = reader1[0].keys()
            fieldnames2 = reader2[0].keys()

            # Add a new column name for the calculated duration
            fieldnames1_new = list(fieldnames1) + ['duration_ratio_2_1']
            fieldnames2_new = list(fieldnames2) + ['duration_ratio_1_2']

            # Prepare data with new columns, skipping division by zero
            for row1, row2 in zip(reader1, reader2):
                duration1 = float(row1['duration']) if row1['duration'] else 0
                duration2 = float(row2['duration']) if row2['duration'] else 0
                
                if duration1 != 0 and duration2 != 0:
                    row1['duration_ratio_2_1'] = duration2 / duration1
                    row2['duration_ratio_1_2'] = duration1 / duration2
                else:
                    row1['duration_ratio_2_1'] = None  # or 'N/A', or simply skip adding this key
                    row2['duration_ratio_1_2'] = None  # or 'N/A', or simply skip adding this key

        # Write the modified data to new files in the new folders
        with open(new_file_path_folder1, mode='w', newline='', encoding='utf-8') as file1, \
             open(new_file_path_folder2, mode='w', newline='', encoding='utf-8') as file2:
            writer1 = csv.DictWriter(file1, fieldnames=fieldnames1_new)
            writer2 = csv.DictWriter(file2, fieldnames=fieldnames2_new)

            writer1.writeheader()
            writer2.writeheader()

            for row1, row2 in zip(reader1, reader2):
                writer1.writerow(row1)
                writer2.writerow(row2)

print("Process completed.")
