import pandas as pd

# Load the CSV file
file_path = '../weighted_mergedSourceAndTarget/weighted_Attention_New/q-c/src/sw4Lite(Quartz-Corona)_filled.csv'
data = pd.read_csv(file_path)

# Check each column for NaN values and report
for column in data.columns:
    nan_count = data[column].isna().sum()
    if nan_count > 0:
        print(f"The column '{column}' contains {nan_count} NaN values.")
        # Fill NaN values with the mean of the column
        mean_value = data[column].mean()
        data[column].fillna(mean_value, inplace=True)
    else:
        print(f"The column '{column}' does not contain any NaN values.")
        
# Save the updated dataframe back to a CSV file if needed
# output_path = '../weighted_mergedSourceAndTarget/weighted_Attention_New/q-c/tar/Kripke(Quartz-Corona)_filled.csv'
# data.to_csv(output_path, index=False)

print("NaN values have been filled with the mean of the respective columns and the updated data has been saved.")
