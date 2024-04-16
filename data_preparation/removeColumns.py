import csv

# Define the input and output file names
input_filename = 'ds_train_n1-1.csv'
output_filename = 'ds_train-n1-1-csv_files/ds_train_n1-1_filtered.csv'

# Define the columns to remove
columns_to_remove = [
    "exec",
    "args",
    "modules",
    "spack_env",
    "exec_path",
    "events",
    "path",
    "PM_MATH_FLOP_CMPL",
    "L1-DCACHE-LOAD-MISSES",
    "L1-ICACHE-LOAD-MISSES",
    "Overhead"
]

with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
    # Create a CSV reader and writer
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=[col for col in reader.fieldnames if col not in columns_to_remove])
    
    # Write the header without the removed columns
    writer.writeheader()
    
    # Write the rows without the removed columns
    for row in reader:
        # Remove the specified columns from the row
        row = {col: value for col, value in row.items() if col not in columns_to_remove}
        writer.writerow(row)

