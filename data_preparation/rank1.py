import csv
import os

# Directory containing the CSV files
directory = 'ml-task1-dataset-csv_files'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        input_filename = os.path.join(directory, filename)
        # Create a new filename with '_filtered' appended before the file extension
        output_filename = os.path.join(directory, os.path.splitext(filename)[0] + '_filtered.csv')
        
        with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            
            writer.writeheader()
            
            for row in reader:
                if row['ranks'] == '1':
                    writer.writerow(row)

print("Filtering complete.")

