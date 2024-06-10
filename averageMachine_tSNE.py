import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio
import sys

# Check for necessary modules and install if missing
try:
    import kaleido
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaleido'])

# Load the CSV file
df = pd.read_csv('../data_preparation/Average_machine_data_with_cores_filledAverage.csv')

# Replace NaN values with the mean of their respective columns
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(df.iloc[:, 1:].mean())

scaler = MinMaxScaler()
df['10'] = scaler.fit_transform(df[['10']])

# Assuming the first column contains the labels
labels = df.iloc[:, 0]
features = df.iloc[:, 1:]

# Simulate a sample_type column for demonstration purposes
df['sample_type'] = ['Sample ' + str(i % 8 + 1) for i in range(len(df))]

# Apply t-SNE
tsne = TSNE(n_components=2,
            perplexity=4,
            early_exaggeration=1,
            learning_rate='auto',
            n_iter=10000,
            random_state=42)
tsne_results = tsne.fit_transform(features)

# Add the t-SNE results to the dataframe
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]

# Available marker symbols for the sample types
marker_symbols = [
    'circle'
]

# Choose a single new marker symbol for all machine types
machine_marker_symbol = 'hexagon'

# Determine the unique sample types and labels present
sample_types = df['sample_type'].unique()
unique_labels = df['machine'].unique()

# Assign colors to each unique label
colors = px.colors.qualitative.Plotly
color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

# Assign a marker to each sample type
marker_map = {sample_type: marker_symbols[i % len(marker_symbols)] for i, sample_type in enumerate(sample_types)}

fig = go.Figure()

# Add traces for sample markers
# for i, sample_type in enumerate(sample_types):
#     fig.add_trace(go.Scatter(
#         x=[None],
#         y=[None],
#         mode='markers',
#         marker=dict(
#             symbol=marker_symbols[i % len(marker_symbols)],
#             size=10,
#             color='white',
#             line=dict(color='black', width=2)
#         ),
#         name=f"Marker for {sample_type}",
#         legendgroup="SampleMarker",
#         showlegend=True
#     ))

# Iterate over each label and sample type to add traces
for label in unique_labels:
    df_label = df[df['machine'] == label]
    for sample_type in sample_types:
        df_subset = df_label[df_label['sample_type'] == sample_type]

        # Add trace to the figure
        fig.add_trace(go.Scatter(
            x=df_subset['tsne-2d-one'],
            y=df_subset['tsne-2d-two'],
            mode='markers',
            marker=dict(
                symbol=marker_map[sample_type],
                size=10,
                color=color_map[label]
            ),
            name=None,
            legendgroup=label,
            showlegend=False
        ))

# Add traces for machine markers
for label in unique_labels:
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol=machine_marker_symbol,
            size=10,
            color=color_map[label]
        ),
        name=label,
        legendgroup="Machine",
        showlegend=True
    ))

# Add a dummy trace to create a line break in the legend
fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    mode='markers',
    marker=dict(size=0),
    name=' ' * 40,  # Add spaces for visual separation
    showlegend=True
))

# Update layout to have separate columns for samples and machines in the legend
fig.update_layout(
    title={
        'text': 't-SNE Visualization of Machine Datasets',
        'y': 0.01,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'bottom',
        'font': {'size': 16}
    },
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    plot_bgcolor='white',
    paper_bgcolor='white',
    shapes=[
        # Line Horizontal
        dict(
            type='line',
            x0=min(tsne_results[:, 0]-1.5),
            y0=-20,
            x1=max(tsne_results[:, 0]+1.2),
            y1=-20,
            line=dict(
                color='black',
                width=2
            )
        ),
        # Line Vertical
        dict(
            type='line',
            x0=-19,
            y0=min(tsne_results[:, 1]-9.8),
            x1=-19,
            y1=max(tsne_results[:, 1]+3),
            line=dict(
                color='black',
                width=2
            )
        )
    ],
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.15,
        xanchor="center",
        x=0.5,
        traceorder="normal"
    ),
    margin=dict(t=150, b=150),
    width=1100,
    height=600
)

# Save the figure to a PDF file
pio.kaleido.scope.mathjax = None
pio.write_image(fig, 'tsne_visualization_10.pdf', format='pdf')

# Display the figure
fig.show()
