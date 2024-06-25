import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load data from CSV
file_path = 'MachineData/Machine.csv'
data = pd.read_csv(file_path)

# Assuming the first column contains text
text_data = data.iloc[:, 0]
rest_of_data = data.iloc[:, 1:]

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number of features
tfidf_features = vectorizer.fit_transform(text_data).toarray()

# Building the autoencoder
input_dim = tfidf_features.shape[1]
encoding_dim = 10  # Embeddings size

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)

# Encoder for extracting embeddings
encoder = Model(input_layer, encoded)

# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(tfidf_features, tfidf_features, epochs=50, batch_size=256, validation_split=0.2)

# Extract embeddings
embeddings = encoder.predict(tfidf_features)

# Combine embeddings with the rest of the data
embeddings_df = pd.DataFrame(embeddings, columns=[f'emb_{i+1}' for i in range(encoding_dim)])
final_data = pd.concat([embeddings_df, rest_of_data.reset_index(drop=True)], axis=1)

# Save the combined data to CSV
final_data.to_csv('combined_data.csv', index=False)

print("Data with embeddings saved to 'combined_data.csv'")
