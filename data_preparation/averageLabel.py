import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the data, possibly skipping the first row if it contains invalid headers
# and setting headers manually if necessary
data = pd.read_csv('C:/Users/ACER/Downloads/Quartz-machine.csv', header=None, skiprows=1)
data.columns = ["app","BR_INS","LD_INS","SR_INS","TOT_INS","L1_LDM","L1_STM","L2_LDM","L2_STM","FP_SINGLE","FP_DOUBLE","ARITH","IO Bytes Read","IO Bytes Written","MEM_WCY","duration"]

# Handle missing values
# Option 1: Fill NaNs with the mean of each column
for column in data.columns:
    if data[column].dtype == float:  # Assuming NaNs are only in float columns
        data[column].fillna(data[column].mean(), inplace=True)

# Option 2: Drop rows with NaNs
# data.dropna(inplace=True)

# Initialize the scaler
scaler = MinMaxScaler()

# Assume all columns except the label column are numeric and need to be normalized
numeric_columns = data.columns.drop('app')  # Adjust the column list as needed

# Normalize the numeric data
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Group the data by the label column and calculate the mean for each group
grouped_data = data.groupby('app').mean()

# Reset the index to turn 'label_column' back into a column
grouped_data = grouped_data.reset_index()

# Save to a new CSV file
grouped_data.to_csv('normalized_averaged_data.csv', index=False)

# Display the result
print(grouped_data)
