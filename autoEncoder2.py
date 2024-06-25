import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import numpy as np

# Load data
data = pd.read_csv('MachineData/Machine.csv')
text_data = data.iloc[:, 0]
rest_of_data = data.iloc[:, 1:]

# Tokenize text data
tokenized_data = [text.split() for text in text_data]

# Load or train a Word2Vec model
# For a pre-trained model (if you have one downloaded)
# model = KeyedVectors.load_word2vec_format('path_to_word2vec.bin', binary=True)

# To train a model on your dataset
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = model.wv

# Convert texts to a fixed size vector by averaging all word vectors in the text
def document_vector(doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word_vectors.key_to_index]
    if not doc:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors[doc], axis=0)

doc_embeddings = np.array([document_vector(doc) for doc in tokenized_data])

# Building the autoencoder
input_dim = doc_embeddings.shape[1]
encoding_dim = 10  # size of the embeddings

input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)
# Encoder for extracting embeddings
encoder = Model(input_layer, encoded)

# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(doc_embeddings, doc_embeddings, epochs=50, batch_size=256, validation_split=0.2)

# Extract embeddings
embeddings = encoder.predict(doc_embeddings)

# Combine embeddings with the rest of the data
embeddings_df = pd.DataFrame(embeddings, columns=[f'emb_{i+1}' for i in range(encoding_dim)])
final_data = pd.concat([embeddings_df, rest_of_data.reset_index(drop=True)], axis=1)

# Save the combined data to CSV
final_data.to_csv('combined_data_with_embeddingsWord2Vec.csv', index=False)

print("Data with embeddings saved to 'combined_data_with_embeddings.csv'")
