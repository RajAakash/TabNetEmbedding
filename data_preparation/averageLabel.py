import pandas as pd

# Load the data from a CSV file
data = pd.read_csv('C:/Users/ACER/Downloads/averageAppEmbeddings.csv')

# Group the data by the label(s) and calculate the mean of each group
grouped_data = data.groupby('app').mean()

# Optionally, reset the index if you want 'label_column' as a regular column
grouped_data = grouped_data.reset_index()

# Save the result to a new CSV file or display it
grouped_data.to_csv('C:/Users/ACER/Downloads/averageDataLabel.csv', index=False)
print(grouped_data)