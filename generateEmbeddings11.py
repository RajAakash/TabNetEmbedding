import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
from tab_network import EmbeddingGenerator

file_path = 'Normalized/Normalized_machine.csv'

df = pd.read_csv(file_path)
X_title=df['machine']
cat_columns = ['machine']  
label_encoders = {col: LabelEncoder() for col in cat_columns}

for col in cat_columns:
    df[col] = label_encoders[col].fit_transform(df[col])
X = df.drop(['target_column'], axis=1, errors='ignore')  

N_machine = len(label_encoders['machine'].classes_)  
print(N_machine) 
E_machine = 10
cat_dims = [N_machine]
cat_idxs = [X.columns.get_loc('machine')]
cat_emb_dims = [E_machine]
# Adjust based on actual features + embedding dimension - 1 (for the encoded column)
input_dim = X.shape[1] 
# input_dim = X.shape[1] + E_machine + E_app - len(cat_columns) 

group_matrix = torch.eye(input_dim)

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
embeddings_df['machine']=X_title
columns = ['machine'] + [col for col in embeddings_df.columns if col != 'machine']
embeddings_df = embeddings_df[columns]
print(embeddings)
embeddings_df.to_csv('Normalized_embeddings_machine.csv', index=False)
print(embeddings.shape)
print(df.shape)
