import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio

# Load the data
# df = pd.read_csv('../averageFilledMachineEmbeddings-method1/finalCombinedConcatenation.csv')
df = pd.read_csv('../Normalized/Embeddings_ApplicationNormalized_Columns/new_Attention_combined.csv')
# df = pd.read_csv('../variance_source_and_target/CombinedVarianceWeights.csv')


# Check for 'kripke', 'sw4lite', 'Laghos', 'minivite', and 'TestDFFT' samples
print("Number of 'kripke' samples:", df[df['Apps'] == 'kripke'].shape[0])
print("Number of 'sw4lite' samples:", df[df['Apps'] == 'sw4lite'].shape[0])
print("Number of 'Laghos' samples:", df[df['Apps'] == 'laghos'].shape[0])
print("Number of 'minivite' samples:", df[df['Apps'] == 'miniVite'].shape[0])
print("Number of 'TestDFFT' samples:", df[df['Apps'] == 'TestDfft'].shape[0])

# Check for 'q-r' and 'q-c' relations
print("Number of 'q-r' samples:", df[df['relation'] == 'q-r'].shape[0])
print("Number of 'q-c' samples:", df[df['relation'] == 'q-c'].shape[0])

# Fill NaN values with the mean of their respective columns
df.iloc[:, 2:] = df.iloc[:, 2:].fillna(0)

# Extract labels and features
labels = df['relation']
features = df.iloc[:, 2:]

# Apply t-SNE
tsne = TSNE(n_components=2,
            perplexity=4,  # Adjusted perplexity value for better visualization
            early_exaggeration=1,
            learning_rate='auto',
            n_iter=10000,
            random_state=42)
tsne_results = tsne.fit_transform(features)

# Add the t-SNE results to the dataframe
df['tsne-2d-one'] = tsne_results[:, 0]
df['tsne-2d-two'] = tsne_results[:, 1]

# Define a unique color and marker for each combination of application and relation
app_relation_map = {
    ('kripke', 'q-r'): {'color': 'red', 'marker': 'square'},
    ('kripke', 'q-c'): {'color': 'darkred', 'marker': 'circle'},
    ('sw4lite', 'q-r'): {'color': 'blue', 'marker': 'square'},
    ('sw4lite', 'q-c'): {'color': 'lightblue', 'marker': 'circle'},
    ('laghos', 'q-r'): {'color': 'green', 'marker': 'square'},
    ('laghos', 'q-c'): {'color': 'lightgreen', 'marker': 'circle'},
    ('miniVite', 'q-r'): {'color': 'purple', 'marker': 'square'},
    ('miniVite', 'q-c'): {'color': 'darkmagenta', 'marker': 'circle'},
    ('TestDfft', 'q-r'): {'color': 'orange', 'marker': 'square'},
    ('TestDfft', 'q-c'): {'color': 'deeppink', 'marker': 'circle'},
}

fig = go.Figure()

# Iterate over each row to add traces with specific markers and colors
for app_relation, style in app_relation_map.items():
    app, relation = app_relation
    subset = df[(df['Apps'] == app) & (df['relation'] == relation)]
    fig.add_trace(go.Scatter(
        x=subset['tsne-2d-one'],
        y=subset['tsne-2d-two'],
        mode='markers',
        marker=dict(
            symbol=style['marker'],
            size=10,
            color=style['color']
        ),
        name=f"{app} {relation}",
        legendgroup=f"{app} {relation}",
        showlegend=True
    ))

# # Add dummy traces for the shapes legend
# fig.add_trace(go.Scatter(
#     x=[None],
#     y=[None],
#     mode='markers',
#     marker=dict(
#         symbol='square',
#         size=10,
#         color='black'
#     ),
#     legendgroup='Shapes',
#     showlegend=True,
#     name='q-r'
# ))

# fig.add_trace(go.Scatter(
#     x=[None],
#     y=[None],
#     mode='markers',
#     marker=dict(
#         symbol='circle',
#         size=10,
#         color='black'
#     ),
#     legendgroup='Shapes',
#     showlegend=True,
#     name='q-c'
# ))

# Update layout to add a proper legend and titles
fig.update_layout(
    title=' ',
    xaxis_title='t-SNE feature 1',
    yaxis_title='t-SNE feature 2',
    width=1200,
    height=800,
    margin=dict(l=40, r=40, t=40, b=100),  # Adding bottom margin for title
    paper_bgcolor='rgba(255,255,255,1)',  # White background
    plot_bgcolor='rgba(255,255,255,1)',  # White plot area
    showlegend=True,
    legend=dict(
        x=0.5,
        y=1.1,
        orientation='h',
        xanchor='center',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,1)',
        bordercolor='black',
        borderwidth=2
    ),
    xaxis=dict(
        showline=True,
        linecolor='black',
        linewidth=2,
    ),
    yaxis=dict(
        showline=True,
        linecolor='black',
        linewidth=2,
    ),
    shapes=[
        dict(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(
                color="black",
                width=2,
            ),
            xref='paper', yref='paper'
        )
    ],
    annotations=[
        dict(
            x=0.5,
            y=-0.15,
            xref='paper',
            yref='paper',
            text='t-SNE Visualization with attention',
            showarrow=False,
            font=dict(
                size=16
            )
        ),
        # Add annotation for shapes legend box
        # Updated annotation for shapes legend box with circle description
        dict(
            x=0.5,
            y=1.07,
            xref='paper',
            yref='paper',
            text='Shapes: q-r - Square, q-c - Circle',
            showarrow=False,
            font=dict(
                size=12
            ),
            bgcolor='rgba(255,255,255,1)',
            bordercolor='black',
            borderwidth=1
        )
    ]
)

# Save the figure to a PDF file
pio.write_image(fig, 'final_book_concat.pdf', format='pdf')

# Display the figure
fig.show()
