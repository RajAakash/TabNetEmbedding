import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def processData(filepath):
    # Read the embeddings
    embeddings_df = pd.read_csv(filepath)

    # Fill NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    embeddings_filled = imputer.fit_transform(embeddings_df)
    embeddings_df_filled = pd.DataFrame(embeddings_filled, columns=embeddings_df.columns)

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_df_filled)
    
    return tsne_results

def createPlot(tsne_results):
    # Plotting
    plt.figure(figsize=(16, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

def main():
    filepath = 'embeddings.csv'  # Path to your CSV file
    tsne_results = processData(filepath)
    createPlot(tsne_results)

if __name__ == "__main__":
    main()

