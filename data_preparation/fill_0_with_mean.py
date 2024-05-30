import pandas as pd

# Load data
df = pd.read_csv('C:/Users/ACER/Downloads/st-AttentionAfterMerge/q-c/tar/kripke-qc.csv')

# Replace 0 with NaN temporarily to calculate mean without considering 0s
df.replace(0, pd.NA, inplace=True)

# Calculate the mean of each column
column_means = df.mean()

# Replace NaNs with the mean of the column
df.fillna(column_means, inplace=True)

# If you want to replace only specific columns, you can do:
# specific_columns = ['column1', 'column2']
# df[specific_columns] = df[specific_columns].replace(0, pd.NA).fillna(column_means)

# Save the modified DataFrame back to a CSV file
df.to_csv('C:/Users/ACER/Downloads/st-AttentionAfterMerge/q-c/tar/kripke-qc.csv', index=False)
