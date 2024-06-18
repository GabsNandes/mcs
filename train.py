import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt

# Carregar o dataset
file_path = 'data/processed/sinan/sinan.parquet'
data = pd.read_parquet(file_path)

# Converter a coluna de datas para o formato datetime
data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])

# Ordenar os dados por ID_UNIDADE e DT_NOTIFIC
data_sorted = data.sort_values(by=['ID_UNIDADE', 'DT_NOTIFIC'])

# Remover a coluna 'Unnamed: 0' se estiver presente
data_sorted = data_sorted.drop(columns=['Unnamed: 0'], errors='ignore')

# Definir a função para criar sequências
def create_sequences(data, sequence_length, features):
    sequences = []
    targets = []
    for unit in data['ID_UNIDADE'].unique():
        unit_data = data[data['ID_UNIDADE'] == unit]
        for i in range(len(unit_data) - sequence_length):
            seq = unit_data.iloc[i:i + sequence_length]
            target = unit_data.iloc[i + sequence_length]
            sequences.append(seq[features].values)
            targets.append(target['CASES'])
    return np.array(sequences), np.array(targets)

# Definir os diferentes conjuntos de recursos
temp_sat_features = ['avg_sat', 'max_sat', 'min_sat']
temp_ws_features = ['avg_ws', 'max_ws', 'min_ws']
rain_sat_features = ['acc_sat']
rain_ws_features = ['acc_ws']
temp_sat_rain_sat_features = temp_sat_features + rain_sat_features
temp_ws_rain_ws_features = temp_ws_features + rain_ws_features
all_features = temp_sat_features + temp_ws_features + rain_sat_features + rain_ws_features

# Criar sequências para cada conjunto de recursos
sequence_length = 7
sequences_temp_sat, targets_temp_sat = create_sequences(data_sorted, sequence_length, temp_sat_features)
sequences_temp_ws, targets_temp_ws = create_sequences(data_sorted, sequence_length, temp_ws_features)
sequences_temp_sat_rain_sat, targets_temp_sat_rain_sat = create_sequences(data_sorted, sequence_length, temp_sat_rain_sat_features)
sequences_temp_ws_rain_ws, targets_temp_ws_rain_ws = create_sequences(data_sorted, sequence_length, temp_ws_rain_ws_features)
sequences_all, targets_all = create_sequences(data_sorted, sequence_length, all_features)

# Normalizar os recursos para cada conjunto
scaler_temp_sat = MinMaxScaler()
scaler_temp_ws = MinMaxScaler()
scaler_temp_sat_rain_sat = MinMaxScaler()
scaler_temp_ws_rain_ws = MinMaxScaler()
scaler_all = MinMaxScaler()

num_sequences_temp_sat, _, num_features_temp_sat = sequences_temp_sat.shape
num_sequences_temp_ws, _, num_features_temp_ws = sequences_temp_ws.shape
num_sequences_temp_sat_rain_sat, _, num_features_temp_sat_rain_sat = sequences_temp_sat_rain_sat.shape
num_sequences_temp_ws_rain_ws, _, num_features_temp_ws_rain_ws = sequences_temp_ws_rain_ws.shape
num_sequences_all, _, num_features_all = sequences_all.shape

sequences_temp_sat_reshaped = sequences_temp_sat.reshape(-1, num_features_temp_sat)
sequences_temp_ws_reshaped = sequences_temp_ws.reshape(-1, num_features_temp_ws)
sequences_temp_sat_rain_sat_reshaped = sequences_temp_sat_rain_sat.reshape(-1, num_features_temp_sat_rain_sat)
sequences_temp_ws_rain_ws_reshaped = sequences_temp_ws_rain_ws.reshape(-1, num_features_temp_ws_rain_ws)
sequences_all_reshaped = sequences_all.reshape(-1, num_features_all)

sequences_temp_sat_normalized = scaler_temp_sat.fit_transform(sequences_temp_sat_reshaped).reshape(num_sequences_temp_sat, sequence_length, num_features_temp_sat)
sequences_temp_ws_normalized = scaler_temp_ws.fit_transform(sequences_temp_ws_reshaped).reshape(num_sequences_temp_ws, sequence_length, num_features_temp_ws)
sequences_temp_sat_rain_sat_normalized = scaler_temp_sat_rain_sat.fit_transform(sequences_temp_sat_rain_sat_reshaped).reshape(num_sequences_temp_sat_rain_sat, sequence_length, num_features_temp_sat_rain_sat)
sequences_temp_ws_rain_ws_normalized = scaler_temp_ws_rain_ws.fit_transform(sequences_temp_ws_rain_ws_reshaped).reshape(num_sequences_temp_ws_rain_ws, sequence_length, num_features_temp_ws_rain_ws)
sequences_all_normalized = scaler_all.fit_transform(sequences_all_reshaped).reshape(num_sequences_all, sequence_length, num_features_all)

# Dividir os dados
split_date = data_sorted['DT_NOTIFIC'].quantile(0.8)
def train_test_split_by_date(sequences, targets, data, split_date):
    train_indices = data['DT_NOTIFIC'] <= split_date
    test_indices = data['DT_NOTIFIC'] > split_date
    
    X_train = sequences[train_indices[:len(sequences)]]
    y_train = targets[train_indices[:len(targets)]]
    X_test = sequences[test_indices[:len(sequences)]]
    y_test = targets[test_indices[:len(targets)]]
    
    return X_train, X_test, y_train, y_test

X_train_temp_sat, X_test_temp_sat, y_train_temp_sat, y_test_temp_sat = train_test_split_by_date(sequences_temp_sat_normalized, targets_temp_sat, data_sorted, split_date)
X_train_temp_ws, X_test_temp_ws, y_train_temp_ws, y_test_temp_ws = train_test_split_by_date(sequences_temp_ws_normalized, targets_temp_ws, data_sorted, split_date)
X_train_temp_sat_rain_sat, X_test_temp_sat_rain_sat, y_train_temp_sat_rain_sat, y_test_temp_sat_rain_sat = train_test_split_by_date(sequences_temp_sat_rain_sat_normalized, targets_temp_sat_rain_sat, data_sorted, split_date)
X_train_temp_ws_rain_ws, X_test_temp_ws_rain_ws, y_train_temp_ws_rain_ws, y_test_temp_ws_rain_ws = train_test_split_by_date(sequences_temp_ws_rain_ws_normalized, targets_temp_ws_rain_ws, data_sorted, split_date)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split_by_date(sequences_all_normalized, targets_all, data_sorted, split_date)

# Função para criar e treinar um modelo LSTM
def train_lstm_model(X_train, y_train, X_test, y_test, input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    return model, history

# Função para criar e treinar um modelo CNN
def train_cnn_model(X_train, y_train, X_test, y_test, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    return model, history

# Treinar os modelos LSTM
model_lstm_temp_sat, history_lstm_temp_sat = train_lstm_model(X_train_temp_sat, y_train_temp_sat, X_test_temp_sat, y_test_temp_sat, (sequence_length, num_features_temp_sat))
model_lstm_temp_ws, history_lstm_temp_ws = train_lstm_model(X_train_temp_ws, y_train_temp_ws, X_test_temp_ws, y_test_temp_ws, (sequence_length, num_features_temp_ws))
model_lstm_temp_sat_rain_sat, history_lstm_temp_sat_rain_sat = train_lstm_model(X_train_temp_sat_rain_sat, y_train_temp_sat_rain_sat, X_test_temp_sat_rain_sat, y_test_temp_sat_rain_sat, (sequence_length, num_features_temp_sat_rain_sat))
model_lstm_temp_ws_rain_ws, history_lstm_temp_ws_rain_ws = train_lstm_model(X_train_temp_ws_rain_ws, y_train_temp_ws_rain_ws, X_test_temp_ws_rain_ws, y_test_temp_ws_rain_ws, (sequence_length, num_features_temp_ws_rain_ws))
model_lstm_all, history_lstm_all = train_lstm_model(X_train_all, y_train_all, X_test_all, y_test_all, (sequence_length, num_features_all))

# Treinar os modelos CNN
model_cnn_temp_sat, history_cnn_temp_sat = train_cnn_model(X_train_temp_sat, y_train_temp_sat, X_test_temp_sat, y_test_temp_sat, (sequence_length, num_features_temp_sat))
model_cnn_temp_ws, history_cnn_temp_ws = train_cnn_model(X_train_temp_ws, y_train_temp_ws, X_test_temp_ws, y_test_temp_ws, (sequence_length, num_features_temp_ws))
model_cnn_temp_sat_rain_sat, history_cnn_temp_sat_rain_sat = train_cnn_model(X_train_temp_sat_rain_sat, y_train_temp_sat_rain_sat, X_test_temp_sat_rain_sat, y_test_temp_sat_rain_sat, (sequence_length, num_features_temp_sat_rain_sat))
model_cnn_temp_ws_rain_ws, history_cnn_temp_ws_rain_ws = train_cnn_model(X_train_temp_ws_rain_ws, y_train_temp_ws_rain_ws, X_test_temp_ws_rain_ws, y_test_temp_ws_rain_ws, (sequence_length, num_features_temp_ws_rain_ws))
model_cnn_all, history_cnn_all = train_cnn_model(X_train_all, y_train_all, X_test_all, y_test_all, (sequence_length, num_features_all))

# Plotar os valores de perda de treinamento e validação para todos os modelos
plt.figure(figsize=(20, 20))

plt.subplot(3, 3, 1)
plt.plot(history_lstm_temp_sat.history['loss'], label='Train')
plt.plot(history_lstm_temp_sat.history['val_loss'], label='Validation')
plt.title('LSTM with Temp Sat Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 2)
plt.plot(history_lstm_temp_ws.history['loss'], label='Train')
plt.plot(history_lstm_temp_ws.history['val_loss'], label='Validation')
plt.title('LSTM with Temp WS Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 3)
plt.plot(history_lstm_temp_sat_rain_sat.history['loss'], label='Train')
plt.plot(history_lstm_temp_sat_rain_sat.history['val_loss'], label='Validation')
plt.title('LSTM with Temp Sat and Rain Sat Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 4)
plt.plot(history_lstm_temp_ws_rain_ws.history['loss'], label='Train')
plt.plot(history_lstm_temp_ws_rain_ws.history['val_loss'], label='Validation')
plt.title('LSTM with Temp WS and Rain WS Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 5)
plt.plot(history_lstm_all.history['loss'], label='Train')
plt.plot(history_lstm_all.history['val_loss'], label='Validation')
plt.title('LSTM with All Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 6)
plt.plot(history_cnn_temp_sat.history['loss'], label='Train')
plt.plot(history_cnn_temp_sat.history['val_loss'], label='Validation')
plt.title('CNN with Temp Sat Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 7)
plt.plot(history_cnn_temp_ws.history['loss'], label='Train')
plt.plot(history_cnn_temp_ws.history['val_loss'], label='Validation')
plt.title('CNN with Temp WS Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 8)
plt.plot(history_cnn_temp_sat_rain_sat.history['loss'], label='Train')
plt.plot(history_cnn_temp_sat_rain_sat.history['val_loss'], label='Validation')
plt.title('CNN with Temp Sat and Rain Sat Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(3, 3, 9)
plt.plot(history_cnn_temp_ws_rain_ws.history['loss'], label='Train')
plt.plot(history_cnn_temp_ws_rain_ws.history['val_loss'], label='Validation')
plt.title('CNN with Temp WS and Rain WS Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()