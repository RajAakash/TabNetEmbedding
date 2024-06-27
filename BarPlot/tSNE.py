import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load and transpose the data
data = pd.read_csv('variance_attention_weights_Quartz.csv', header=0, index_col=0)
#data = pd.read_csv('attention_weights_Quartz.csv', header=0, index_col=0).transpose()
data_filled = data.fillna(data.mean())

# Apply t-SNE
tsne = TSNE(n_components=2,perplexity=1, random_state=42)
transformed_data = tsne.fit_transform(data_filled)

# Define colors for each label
colors = ['red', 'blue', 'green']

# Plot
plt.figure(figsize=(8, 6))
for i, txt in enumerate(data.columns):
    plt.scatter(transformed_data[i, 0], transformed_data[i, 1], color=colors[i], label=txt)

# Add annotations and labels
for i, txt in enumerate(data.columns):
    plt.annotate(txt, (transformed_data[i, 0], transformed_data[i, 1]))

plt.xlabel('t-SNE Feature 1')
plt.xticks([])
plt.yticks([])
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization of Attention Weights of Attention Weights')
plt.legend()
plt.grid(True)
plt.show()
