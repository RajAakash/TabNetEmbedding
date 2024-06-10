from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assuming 'data' is your input data matrix
# Convert the numpy array to a pandas DataFrame for easier NaN handling
df = pd.read_csv('embeddings.csv')

# Check for NaN values and fill them with the mean of their respective columns
df.fillna(df.mean(), inplace=True)

# Convert back to numpy array if needed
cleaned_data = df.to_numpy()

# Perform t-SNE on the cleaned data
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(cleaned_data)

# Assuming we have a way to generate color labels for our data points
# This is a placeholder step, adjust according to your data
colors = ['red' if i < 15 else 'blue' for i in range(cleaned_data.shape[0])]

# Plot t-SNE results
plt.figure(figsize=(8, 5))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.5)
plt.title('t-SNE visualization of cleaned data')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
