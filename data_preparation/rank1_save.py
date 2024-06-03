import pandas as pd

# Load the CSV file
input_file = '../../../../Documents/Laghos-on-Quartz.csv'  # replace with your file name
df = pd.read_csv(input_file)

# Filter rows where rank is 1
filtered_df = df[df['rank'] == 1]

# Save the filtered data to a new CSV file
output_file = 'nLaghos-on-Quartz_rank1.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
