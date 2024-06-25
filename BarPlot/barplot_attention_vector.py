import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file, assuming no header and data is in rows
data = pd.read_csv('weighted_embeddings_Corona.csv', header=None)

# Transpose the data to turn rows into columns
data = data.transpose()

# Rename columns appropriately after the transpose
data.columns = ['Index', 'Value']

# Plotting
plt.figure(figsize=(10, 5))  # Adjust the size of the plot as needed
plt.bar(data['Index'], data['Value'], color='blue')  # Create a bar plot
plt.grid(False) 
plt.xticks([])
plt.yticks([])
plt.grid(axis='y')
plt.xlabel('Index')  # Label for the x-axis
plt.ylabel('Attention value')  # Label for the y-axis
plt.title('Bar Plot of Quartz Attention')  # Title of the plot
plt.grid(True)  # Enable grid for better readability
plt.show()  # Display the plot
