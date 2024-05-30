import pandas as pd

# Load the data from the CSV file
file_path = '../machineEmbedding/machineEmbeddings_10.csv'
data = pd.read_csv(file_path)

# Group the data by 'Machine' and calculate the mean for each column
average_data = data.groupby('machine').mean()

# Print the resulting DataFrame with averages
print(average_data)
average_data.to_csv('Average_machine_data_with_cores.csv')
