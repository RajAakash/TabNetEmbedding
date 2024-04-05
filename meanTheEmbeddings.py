import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('embeddings.csv')

# Calculate the mean of each column
means = df.mean()

# Fill NaN values with the mean of their respective columns
df_filled = df.fillna(means)

# If you want to save the modified DataFrame back to a CSV
df_filled.to_csv('embeddings_mean.csv', index=False)