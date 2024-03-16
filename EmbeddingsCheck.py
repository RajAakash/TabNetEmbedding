import pandas as pd

# Assuming df is your DataFrame after reading the CSV
df = pd.read_csv("Machine.csv")

# Convert 'machine' to categorical codes correctly ensuring it starts from 0
df['machine'] = df['machine'].astype('category').cat.codes

# Now, ensure 'core' is treated correctly. If 'core' is categorical, encode it similarly:
# If 'core' is already numerical and simply represents different machines, you might not need to encode it.

# Verify max values for categorical features do not exceed embedding sizes
print("Max 'machine' code:", df['machine'].max())
# Repeat for 'core' if it's encoded

