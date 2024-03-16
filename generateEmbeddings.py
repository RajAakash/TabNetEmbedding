import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from tab_network import EmbeddingGenerator

file_path = 'Machine.csv'

df = pd.read_csv(file_path)
# Encode categorical columns
cat_columns = ['machine','core']  # Add more categorical column names if you have any
label_encoders = {col: LabelEncoder() for col in cat_columns}

for col in cat_columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# Assuming all other columns are continuous and do not require normalization for this example
# If normalization is required, you can use sklearn's StandardScaler or MinMaxScaler

# Separate features and possibly targets if you have them
X = df.drop(['target_column'], axis=1, errors='ignore')  # Drop target column if exists

# Use the adjusted parameters based on your dataset
N_machine = len(label_encoders['machine'].classes_)  # Number of unique machines
N_core = len(label_encoders['core'].classes_)  
E_machine = 10   # Example embedding dimension
E_core = 5
cat_dims = [N_machine, N_core]
cat_idxs = [X.columns.get_loc('machine'), X.columns.get_loc('core')]
cat_emb_dims = [E_machine, E_core]
# Adjust based on actual features + embedding dimension - 1 (for the encoded column)
input_dim = X.shape[1] 
# input_dim = X.shape[1] + E_machine + E_core - len(cat_columns) 

# Dummy group matrix for demonstration; adjust as needed
group_matrix = torch.eye(input_dim)

# Instantiate the EmbeddingGenerator
embedding_generator = EmbeddingGenerator(
    input_dim=input_dim,
    cat_dims=cat_dims,
    cat_idxs=cat_idxs,
    cat_emb_dims=cat_emb_dims,
    group_matrix=group_matrix
)
X_tensor = torch.tensor(X.values, dtype=torch.float32)
embeddings = embedding_generator.forward(X_tensor)
embeddings_df = pd.DataFrame(embeddings.detach().numpy())
print(embeddings)
embeddings_df.to_csv('embeddings.csv', index=False)
print(embeddings.shape)
print(df.shape)