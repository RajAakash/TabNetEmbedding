import pandas as pd

# Load your CSV file
df = pd.read_csv('Average_machine_data_with_cores.csv')

# Remove the 'machine' column and store it separately
machine_column = df['machine']
df = df.drop('machine', axis=1)

# Calculate the mean of each column, skipping NaN values
column_means = df.mean()

# Fill NaN values with the mean of each column
df_filled = df.fillna(column_means)

# Reattach the 'machine' column at the end of the DataFrame
df_filled['machine'] = machine_column
print(machine_column)

# Save the modified DataFrame back to a CSV file
df_filled.to_csv('Average_machine_data_with_cores_filledAverage.csv', index=False)