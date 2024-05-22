import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
file_path = 'data/processed/sinan/sinan.parquet'
data = pd.read_parquet(file_path)

# Drop the ID_AGRAVO column
if 'ID_AGRAVO' in data.columns:
    data = data.drop(columns=['ID_AGRAVO'])

# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Convert the date column to datetime format and sort by date
data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
data = data.dropna(subset=['DT_NOTIFIC'])  # Drop rows where date conversion failed
data = data.sort_values('DT_NOTIFIC')

# Check for NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Group the data by ID_UNIDADE
grouped = data.groupby('ID_UNIDADE')

# Function to prepare sequences
def create_sequences(X, y, seq_length=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

# Prepare a dictionary to store models and scalers for each ID_UNIDADE
models = {}
scalers = {}

for id_unidade, group in grouped:
    # Drop unnecessary columns and separate features and target
    X = group.drop(columns=['CASES', 'DT_NOTIFIC', 'ID_UNIDADE'])
    y = group['CASES']
    
    # Normalize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Prepare sequences
    seq_length = 10
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
    
    # Debugging prints to inspect shapes
    print(f'ID_UNIDADE: {id_unidade}')
    print(f'X_train shape: {X_train.shape}')
    print(f'X_train_seq shape: {X_train_seq.shape}')
    
    if X_train_seq.shape[0] == 0 or X_train_seq.shape[1] == 0 or X_train_seq.shape[2] == 0:
        print(f'Skipping ID_UNIDADE: {id_unidade} due to insufficient data after sequencing.')
        continue
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, X_train_seq.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1)
    
    # Store the model and scaler
    models[id_unidade] = model
    scalers[id_unidade] = scaler
    
    # Evaluate the model
    loss = model.evaluate(X_test_seq, y_test_seq)
    print(f'ID_UNIDADE: {id_unidade}, Test Loss: {loss}')
    
    # Predict the cases
    y_pred = model.predict(X_test_seq)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_seq, label='True Cases')
    plt.plot(y_pred, label='Predicted Cases')
    plt.title(f'ID_UNIDADE: {id_unidade}')
    plt.legend()
    plt.show()
