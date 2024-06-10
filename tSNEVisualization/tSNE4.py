import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming `data` is your DataFrame loaded with the embeddings
# Let's first deal with NaN values
data=pd.read_csv('embeddings.csv')
data.fillna(data.mean(), inplace=True)

# Create a label for each unique combination in the first 15 columns
# This creates a unique integer label for each unique combination of values in the first 15 columns
data['cluster_label'] = data.iloc[:, :15].astype(str).agg('-'.join, axis=1).factorize()[0]

# Now, let's perform t-SNE on the entire dataset excluding our new 'cluster_label' column
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data.drop(columns=['cluster_label']))

# Plotting
plt.figure(figsize=(8, 5))

# Scatter plot, with colors based on 'cluster_label'
# cmap='viridis' can be changed to any matplotlib colormap
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data['cluster_label'], cmap='viridis', alpha=0.5)

# Creating a colorbar
plt.colorbar(scatter)

plt.title('t-SNE visualization of embeddings with similar first 15 columns clustered')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
