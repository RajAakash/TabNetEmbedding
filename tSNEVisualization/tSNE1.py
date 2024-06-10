import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import cycle

# Load the CSV file
df = pd.read_csv('embeddings1.csv')

# Replace NaN values with the mean of their respective columns
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(df.iloc[:, 1:].mean())

# Assuming the first column contains the labels
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

# Apply t-SNE
tsne = TSNE(n_components=2, 
            perplexity=8, 
            early_exaggeration=1,
            learning_rate='auto',
            n_iter=1000, 
            random_state=42
        )
tsne_results = tsne.fit_transform(features)

# Plotting setup
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)

# Manually specified sharp colors
sharp_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
    '#ffffff', '#000000'
]

if len(unique_labels) > len(sharp_colors):
    color_cycle = cycle(sharp_colors)
    label_to_color = {label: next(color_cycle) for label in unique_labels}
else:
    label_to_color = dict(zip(unique_labels, sharp_colors))

# Define a cycle of markers
markers = cycle(['o', 'x', '^', '>', '<', 'p', '*', 'h', 'H', '+', 'D', 'd', '|', '_'])

added_to_legend = {label: False for label in unique_labels}

for i in range(len(df)):
    marker = next(markers)  
    label = labels.iloc[i]
    color = label_to_color[label]
    
    if not added_to_legend[label]:
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], label=label, marker=marker, color=color)
        added_to_legend[label] = True  
    else:
        plt.scatter(tsne_results[i, 0], tsne_results[i, 1], marker=marker, color=color)

plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE plot of the dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.tight_layout()  
plt.show()

