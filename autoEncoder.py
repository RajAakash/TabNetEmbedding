import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Load data from CSV
file_path = 'MachineData/Machine.csv'
data = pd.read_csv(file_path)

# Convert all columns to numeric, assuming the first column will be skipped as it's a label
data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean of each column
data.iloc[:, 1:] = data.iloc[:, 1:].fillna(data.iloc[:, 1:].mean())

# Ensure there are no NaN values left
print(data.isnull().sum())

# Separate the first column as labels and use the rest as features
labels = data.iloc[:, 0]  # Assuming the first column is the label
features = data.iloc[:, 1:]  # The rest are features

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Building the autoencoder
input_layer = Input(shape=(scaled_features.shape[1],))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(10, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(scaled_features.shape[1], activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Encoder model to extract embeddings
encoder = Model(input_layer, bottleneck)
embeddings = encoder.predict(scaled_features)
embeddings.to_csv('embeddings_from_autoencoder.csv')
print(embeddings)
