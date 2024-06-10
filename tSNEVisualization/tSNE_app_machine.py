import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('../data_preparation/Average_machine_data_with_cores_filledAverage.csv')
df2 = pd.read_csv('../appEmbeddings/appEmbeddings15_10.csv')

# Replace NaN values with the mean of their respective columns
df1.iloc[:, 1:] = df1.iloc[:, 1:].fillna(df1.iloc[:, 1:].mean())
df2.iloc[:, 1:] = df2.iloc[:, 1:].fillna(df2.iloc[:, 1:].mean())

# Normalize the features in both datasets
scaler = MinMaxScaler()
df1.iloc[:, 1:] = scaler.fit_transform(df1.iloc[:, 1:])
df2.iloc[:, 1:] = scaler.fit_transform(df2.iloc[:, 1:])

# Perform t-SNE separately on each dataset
tsne1 = TSNE(n_components=2, 
             perplexity=4, 
             early_exaggeration=1, 
             learning_rate='auto', 
             n_iter=10000, 
             random_state=42
            )
tsne_results1 = tsne1.fit_transform(df1.iloc[:, 1:])

tsne2 = TSNE(n_components=2, 
             perplexity=20, 
             early_exaggeration=1, 
             learning_rate='auto', 
             n_iter=10000, 
             random_state=42
            )
tsne_results2 = tsne2.fit_transform(df2.iloc[:, 1:])

# Create separate dataframes for the t-SNE results
df1_tsne = pd.DataFrame(tsne_results1, columns=['tsne-2d-one', 'tsne-2d-two'])
df1_tsne['label'] = df1.iloc[:, 0]
df1_tsne['dataset'] = 'df1'
df1_tsne['sample_type'] = ['Sample ' + str(i % 8 + 1) for i in range(len(df1))]

df2_tsne = pd.DataFrame(tsne_results2, columns=['tsne-2d-one', 'tsne-2d-two'])
df2_tsne['label'] = df2.iloc[:, 0]
df2_tsne['dataset'] = 'df2'

# Combine the t-SNE results
combined_tsne_df = pd.concat([df1_tsne, df2_tsne], ignore_index=True)

# Available marker symbols for df1
marker_symbols = ['square']

# Assign a marker to each sample type in df1
marker_map_df1 = {sample_type: marker_symbols[i % len(marker_symbols)] for i, sample_type in enumerate(df1_tsne['sample_type'].unique())}

# Determine the unique labels present
unique_labels = combined_tsne_df['label'].unique()

# Generate distinct colors using matplotlib's 'tab20' palette
colors = plt.get_cmap('tab20')
color_map = {label: f'rgb({int(colors(i/len(unique_labels))[0]*255)}, {int(colors(i/len(unique_labels))[1]*255)}, {int(colors(i/len(unique_labels))[2]*255)})' for i, label in enumerate(unique_labels)}

fig = go.Figure()

# # Add a dummy trace for each marker symbol to show it in the legend
# for i, sample_type in enumerate(df1_tsne['sample_type'].unique()):
#     fig.add_trace(go.Scatter(
#         x=[None],
#         y=[None],
#         mode='markers',
#         marker=dict(
#             symbol=marker_symbols[i % len(marker_symbols)],
#             size=10
#         ),
#         name=f"Marker for {sample_type}",
#         legendgroup="Marker",  # Group all marker symbols under the name "Marker"
#         showlegend=True
#     ))

# Track labels added to legend
labels_added_to_legend = set()

# Iterate over each label and dataset to add traces
for label in unique_labels:
    # Add traces for df1
    df1_subset = combined_tsne_df[(combined_tsne_df['label'] == label) & (combined_tsne_df['dataset'] == 'df1')]
    for sample_type in df1_subset['sample_type'].unique():
        df1_sample_type_subset = df1_subset[df1_subset['sample_type'] == sample_type]

        # Determine if this is the first instance of the label
        first_instance_of_label = label not in labels_added_to_legend
        if first_instance_of_label:
            labels_added_to_legend.add(label)

        fig.add_trace(go.Scatter(
            x=df1_sample_type_subset['tsne-2d-one'],
            y=df1_sample_type_subset['tsne-2d-two'],
            mode='markers',
            marker=dict(
                symbol=marker_map_df1[sample_type],
                size=10,
                color=color_map[label]
            ),
            name=label if first_instance_of_label else None,  # Only add label name for the first instance
            legendgroup=label,  # Group all instances under the same label
            showlegend=first_instance_of_label  # Show legend only for the first instance
        ))

    # Add traces for df2
    df2_subset = combined_tsne_df[(combined_tsne_df['label'] == label) & (combined_tsne_df['dataset'] == 'df2')]

    fig.add_trace(go.Scatter(
        x=df2_subset['tsne-2d-one'],
        y=df2_subset['tsne-2d-two'],
        mode='markers',
        marker=dict(
            symbol='circle',
            size=10,
            color=color_map[label]
        ),
        name=label if label not in labels_added_to_legend else None,  # Only add label name for the first instance
        legendgroup=label,  # Group all instances under the same label
        showlegend=label not in labels_added_to_legend  # Show legend only for the first instance
    ))
    labels_added_to_legend.add(label)

# Update layout to add a proper legend and titles as before
fig.update_layout(
    title='Overlayed t-SNE Visualization with Machines and Applications',
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    legend_title="Legend",
    width=1100,
    height=600
)

# Save the figure as a PDF
fig.write_image("tsne_application_machine.pdf", format="pdf")
fig.show()