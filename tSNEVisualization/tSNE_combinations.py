import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Load the data
df = pd.read_csv('../sourceFolderEmbeddings/final_book_concat.csv')

# Check for 'kripke', 'sw4lite', 'Laghos', 'minivite', and 'TestDFFT' samples
print("Number of 'kripke' samples:", df[df['Apps'] == 'kripke'].shape[0])
print("Number of 'sw4lite' samples:", df[df['Apps'] == 'sw4lite'].shape[0])
print("Number of 'Laghos' samples:", df[df['Apps'] == 'Laghos'].shape[0])
print("Number of 'minivite' samples:", df[df['Apps'] == 'minivite'].shape[0])
print("Number of 'TestDFFT' samples:", df[df['Apps'] == 'TestDFFT'].shape[0])

# Fill NaN values with the mean of their respective columns
df.iloc[:, 2:] = df.iloc[:, 2:].fillna(df.iloc[:, 2:].mean())

# Extract labels and features
labels = df['relation']
features = df.iloc[:, 2:]

# Apply t-SNE
tsne = TSNE(n_components=2,
            perplexity=1,  # Adjusted perplexity value for better visualization
            early_exaggeration=1,
            learning_rate='auto',
            n_iter=10000,
            random_state=42)
tsne_results = tsne.fit_transform(features)

# Add the t-SNE results to the dataframe
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]

# Define colors for applications
color_map = {
    'kripke': 'red',
    'sw4lite': 'blue',
    'Laghos': 'green',
    'minivite': 'purple',
    'TestDFFT': 'orange'
}

# Define markers for relations
marker_map = {
    'q-r': 'square',
    'q-c': 'circle'
}

fig = go.Figure()

# Iterate over each row to add traces with specific markers and colors
for _, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['tsne-2d-one']],
        y=[row['tsne-2d-two']],
        mode='markers',
        marker=dict(
            symbol=marker_map[row['relation']],
            size=10,
            color=color_map[row['Apps']]
        ),
        name=f"{row['Apps']} {row['relation']}",
        showlegend=False  # Legends will be added separately
    ))

# Add dummy traces for the legend
for relation in marker_map.keys():
    for app in color_map.keys():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                symbol=marker_map[relation],
                size=10,
                color=color_map[app]
            ),
            name=f"{app} {relation}",
            legendgroup=f"{app} {relation}",
            showlegend=True
        ))

# Update layout to add a proper legend and titles
fig.update_layout(
    title='t-SNE Visualization with Relation Markers and Application Colors',
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    legend_title="Legend",
    width=1100,
    height=600
)

# Save the figure to a PDF file
pio.write_image(fig, 'final_book_concat.pdf', format='pdf')

# Display the figure
fig.show()
