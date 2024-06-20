import pandas as pd

# Load the CSV file
df = pd.read_csv('MachineData/grouped_means.csv')

# Assume the first column is the label column; let's temporarily remove it
labels = df.iloc[:, 0]  # This stores the first column (label column)
df_data = df.iloc[:, 1:]  # This stores the rest of the data

# Calculate the mean of each column in df_data
means = df_data.mean()

# Fill NaN values in df_data with the mean of their respective columns
df_filled_data = df_data.fillna(means)

# Reinsert the label column back into the DataFrame
df_filled = pd.concat([labels, df_filled_data], axis=1)

# Save the modified DataFrame back to a CSV file
df_filled.to_csv('MachineData/embeddings_mean_and_AverageFilled.csv', index=False)

print("DataFrame with NaN values filled with column means, excluding labels, has been saved.")
