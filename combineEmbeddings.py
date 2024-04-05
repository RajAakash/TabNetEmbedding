import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('embeddings_mean.csv')

# Define a custom aggregation function to concatenate values horizontally
def concatenate_rows(series):
    return pd.Series(series.dropna().unique())

# Group by the first 10 columns and apply the custom aggregation function
grouped = df.groupby(list(df.columns[:10])).agg(concatenate_rows)

# Reset index if you want the grouping columns back as regular columns
result = grouped.reset_index()

# If you want to save the result to a CSV
result.to_csv('combined_file.csv', index=False)