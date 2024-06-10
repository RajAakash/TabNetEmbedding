import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio

# Load the data
df = pd.read_csv('../sourceFolderEmbeddings/final_book_concat.csv')

# Fill NaN values with the mean of their respective columns
df.iloc[:, 2:] = df.iloc[:, 2:].fillna(df.iloc[:, 2:].mean())

# Filter the data for 'kripke' application and 'q-r' and 'q-c' relations
df_filtered = df[(df['Apps'] == 'kripke') & (df['relation'].isin(['q-r', 'q-c']))]

# Extract labels and features
labels = df_filtered['relation']
features = df_filtered.iloc[:, 2:]

# Apply t-SNE
tsne = TSNE(n_components=2,
            perplexity=15,
            early_exaggeration=1,
            learning_rate='auto',
            n_iter=10000,
            random_state=42)
tsne_results = tsne.fit_transform(features)

# Add the t-SNE results to the dataframe
df_filtered['tsne-2d-one'] = tsne_results[:, 0]
df_filtered['tsne-2d-two'] = tsne_results[:, 1]

# Assign colors to each unique relation
unique_relations = df_filtered['relation'].unique()
colors = px.colors.qualitative.Plotly
color_map = {relation: colors[i % len(colors)] for i, relation in enumerate(unique_relations)}

# Create the plot
fig = go.Figure()

# Add traces for each relation
for relation in unique_relations:
    df_subset = df_filtered[df_filtered['relation'] == relation]
    fig.add_trace(go.Scatter(
        x=df_subset['tsne-2d-one'],
        y=df_subset['tsne-2d-two'],
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=10,
            color=color_map[relation]
        ),
        name=relation,
        legendgroup=relation,
        showlegend=True
    ))

# Update layout to add a proper legend and titles
fig.update_layout(
    title='t-SNE Visualization for Kripke with Relation Colors (q-r and q-c)',
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    legend_title="Legend",
    width=1100,
    height=600
)

# Save the figure to a PDF file
pio.write_image(fig, 'tsne_visualization_kripke_qr_qc.pdf', format='pdf')

# Display the figure
fig.show()
