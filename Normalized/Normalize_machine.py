from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def min_max_normalize(data):
    """
    Applies Min-Max normalization to scale the data between 0 and 1 for each feature.

    Parameters:
        data (pd.DataFrame): Input data to be normalized.
    
    Returns:
        pd.DataFrame: Data scaled to the range [0, 1] across all features.
    """
    # Separate the first columns as labels
    labels = data.iloc[:, :1]

    # Select the rest of the columns for scaling
    data_to_scale = data.iloc[:, 1:]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    normalized_data = scaler.fit_transform(data_to_scale)
    
    # Convert the array back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=data_to_scale.columns, index=data.index)
    
    # Prepend the label columns to the beginning of the DataFrame
    normalized_df = pd.concat([labels, normalized_df], axis=1)
    
    return normalized_df

# Example usages
data = pd.read_csv('Machine.csv')
normalized_data = min_max_normalize(data)
normalized_data.to_csv('Normalized_machine.csv')
# print(normalized_data)
