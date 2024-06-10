import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Load the data
df = pd.read_csv('../sourceFolderEmbeddings/final_book_concat.csv')

# Check for 'kripke' and 'sw4lite' samples
print("Number of 'kripke' samples:", df[df['Apps'] == 'kripke'].shape[0])
print("Number of 'sw4lite' samples:", df[df['Apps'] == 'sw4lite'].shape[0])
print("Number of 'Laghos' samples:", df[df['Apps'] == 'Laghos'].shape[0])

# Fill NaN values with the mean of their respective columns
df.iloc[:, 2:] = df.iloc[:, 2:].fillna(df.iloc[:, 2:].mean())

# Filter the data for 'kripke' and 'sw4lite' applications and 'q-r' and 'q-c' relations
df_filtered = df[(df['Apps'].isin(['kripke', 'sw4lite'])) & (df['relation'].isin(['q-r', 'q-c']))]

# Check the filtered data
print("Filtered DataFrame shape:", df_filtered.shape)
print(df_filtered['Apps'].value_counts())
print(df_filtered['relation'].value_counts())

# Extract labels and features
labels = df_filtered['relation']
features = df_filtered.iloc[:, 2:]

# Apply t-SNE with adjusted perplexity
tsne = TSNE(n_components=2,
            perplexity=30,  # Adjusted perplexity value for better visualization
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

# Assign marker symbols to each application
unique_apps = df_filtered['Apps'].unique()
marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'pentagon']
marker_map = {app: marker_symbols[i % len(marker_symbols)] for i, app in enumerate(unique_apps)}

fig = go.Figure()

# Add a dummy trace for each marker symbol to show it in the legend
for i, app in enumerate(unique_apps):
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol=marker_symbols[i % len(marker_symbols)],
            size=10
        ),
        name=f"Marker for {app}",
        legendgroup="Marker",  # Group all marker symbols under the name "Marker"
        showlegend=True
    ))

# Track relations added to legend
relations_added_to_legend = set()

# Iterate over each relation and application to add traces
for relation in unique_relations:
    df_relation = df_filtered[df_filtered['relation'] == relation]
    for app in unique_apps:
        df_subset = df_relation[df_relation['Apps'] == app]

        # Determine if this is the first instance of the relation
        first_instance_of_relation = relation not in relations_added_to_legend
        if first_instance_of_relation:
            relations_added_to_legend.add(relation)

        # Add trace to the figure
        fig.add_trace(go.Scatter(
            x=df_subset['tsne-2d-one'],
            y=df_subset['tsne-2d-two'],
            mode='markers',
            marker=dict(
                symbol=marker_map[app],
                size=10,
                color=color_map[relation]
            ),
            name=relation if first_instance_of_relation else None,  # Only add relation name for the first instance
            legendgroup=relation,  # Group all instances under the same relation
            showlegend=first_instance_of_relation  # Show legend only for the first instance
        ))

# Update layout to add a proper legend and titles
fig.update_layout(
    title='t-SNE Visualization with Relation Colors and Application Markers (Kripke and Sw4lite)',
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    legend_title="Legend",
    width=1100,
    height=600
)

# Save the figure to a PDF file
pio.write_image(fig, 'tsne_visualization_kripke_sw4lite.pdf', format='pdf')

# Display the figure
fig.show()
