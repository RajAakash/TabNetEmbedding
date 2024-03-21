import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the embeddings data from the provided text (pre-processed into CSV format for loading convenience)
# The data seems to be provided as a list with a length of 78 for each embedding
# The actual embedding vectors are comma-separated and the first value seems to be an index
# For convenience, the data was pre-processed into a CSV with 78 columns (first one being the index)
# The file was saved with the name "embeddings.csv"

# Function to process the embeddings and perform t-SNE
def process_embeddings(file_path):
    # Read the CSV file, assuming the first column is an index
    embeddings_df = pd.read_csv(file_path, index_col=0)

    # Check if there are any missing values and fill them with the mean of the column
    if embeddings_df.isnull().values.any():
        embeddings_df.fillna(embeddings_df.mean(), inplace=True)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings_df)
    
    return tsne_results

# Process the embeddings
file_path = 'embeddings.csv' # Update with the correct path if necessary
tsne_results = process_embeddings(file_path)

# Number of embeddings
num_embeddings = tsne_results.shape[0]

# Generate color labels (assumption: each set of 15 embeddings should have the same color, and there are 76 embeddings)
num_colors = num_embeddings // 15 + (num_embeddings % 15 > 0)
colors = np.tile(np.arange(num_colors), 15)[:num_embeddings]

# Plot t-SNE results
plt.figure(figsize=(8, 5))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.title('t-SNE visualization of embeddings')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
