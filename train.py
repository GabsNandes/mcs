import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt

# Carregar o dataset
csv_file_path = 'path_to_your_csv_file.csv'
data = pd.read_csv(csv_file_path)

# Converter a coluna de datas para o formato datetime
data['DT_NOTIFIC'] = pd.to_datetime(data['DT_NOTIFIC'])

# Ordenar os dados por ID_UNIDADE e DT_NOTIFIC
data_sorted = data.sort_values(by=['ID_UNIDADE', 'DT_NOTIFIC'])

# Remover a coluna 'Unnamed: 0' se estiver presente
data_sorted = data_sorted.drop(columns=['Unnamed: 0'], errors='ignore')

# Função para criar sequências de um determinado comprimento a partir dos dados
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
all_features = ['avg_sat', 'max_sat', 'min_sat', 'avg_ws', 'max_ws', 'min_ws', 'lat', 'lon']
sat_features = ['avg_sat', 'max_sat', 'min_sat']
ws_features = ['avg_ws', 'max_ws', 'min_ws']

# Criar sequências para cada conjunto de recursos
sequence_length = 7
sequences_all, targets_all = create_sequences(data_sorted, sequence_length, all_features)
sequences_sat, targets_sat = create_sequences(data_sorted, sequence_length, sat_features)
sequences_ws, targets_ws = create_sequences(data_sorted, sequence_length, ws_features)

# Normalizar os recursos para cada conjunto
scaler_all = MinMaxScaler()
scaler_sat = MinMaxScaler()
scaler_ws = MinMaxScaler()

num_sequences_all, _, num_features_all = sequences_all.shape
num_sequences_sat, _, num_features_sat = sequences_sat.shape
num_sequences_ws, _, num_features_ws = sequences_ws.shape

sequences_all_reshaped = sequences_all.reshape(-1, num_features_all)
sequences_sat_reshaped = sequences_sat.reshape(-1, num_features_sat)
sequences_ws_reshaped = sequences_ws.reshape(-1, num_features_ws)

sequences_all_normalized = scaler_all.fit_transform(sequences_all_reshaped).reshape(num_sequences_all, sequence_length, num_features_all)
sequences_sat_normalized = scaler_sat.fit_transform(sequences_sat_reshaped).reshape(num_sequences_sat, sequence_length, num_features_sat)
sequences_ws_normalized = scaler_ws.fit_transform(sequences_ws_reshaped).reshape(num_sequences_ws, sequence_length, num_features_ws)

# Determinar o ponto de divisão com base nas datas (ex.: 80% para treinamento, 20% para teste)
split_date = data_sorted['DT_NOTIFIC'].quantile(0.8)

# Função para dividir os dados de treino e teste com base na data
def train_test_split_by_date(sequences, targets, data, split_date):
    train_indices = data['DT_NOTIFIC'] <= split_date
    test_indices = data['DT_NOTIFIC'] > split_date
    
    X_train = sequences[train_indices[:len(sequences)]]
    y_train = targets[train_indices[:len(targets)]]
    X_test = sequences[test_indices[:len(sequences)]]
    y_test = targets[test_indices[:len(targets)]]
    
    return X_train, X_test, y_train, y_test

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split_by_date(sequences_all_normalized, targets_all, data_sorted, split_date)
X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split_by_date(sequences_sat_normalized, targets_sat, data_sorted, split_date)
X_train_ws, X_test_ws, y_train_ws, y_test_ws = train_test_split_by_date(sequences_ws_normalized, targets_ws, data_sorted, split_date)

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
model_lstm_all, history_lstm_all = train_lstm_model(X_train_all, y_train_all, X_test_all, y_test_all, (sequence_length, num_features_all))
model_lstm_sat, history_lstm_sat = train_lstm_model(X_train_sat, y_train_sat, X_test_sat, y_test_sat, (sequence_length, num_features_sat))
model_lstm_ws, history_lstm_ws = train_lstm_model(X_train_ws, y_train_ws, X_test_ws, y_test_ws, (sequence_length, num_features_ws))

# Treinar os modelos CNN
model_cnn_all, history_cnn_all = train_cnn_model(X_train_all, y_train_all, X_test_all, y_test_all, (sequence_length, num_features_all))
model_cnn_sat, history_cnn_sat = train_cnn_model(X_train_sat, y_train_sat, X_test_sat, y_test_sat, (sequence_length, num_features_sat))
model_cnn_ws, history_cnn_ws = train_cnn_model(X_train_ws, y_train_ws, X_test_ws, y_test_ws, (sequence_length, num_features_ws))

# Plotar os valores de perda de treinamento e validação para todos os modelos
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(history_lstm_all.history['loss'], label='Train')
plt.plot(history_lstm_all.history['val_loss'], label='Validation')
plt.title('LSTM with All Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(2, 3, 2)
plt.plot(history_lstm_sat.history['loss'], label='Train')
plt.plot(history_lstm_sat.history['val_loss'], label='Validation')
plt.title('LSTM with Satellite Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(2, 3, 3)
plt.plot(history_lstm_ws.history['loss'], label='Train')
plt.plot(history_lstm_ws.history['val_loss'], label='Validation')
plt.title('LSTM with Weather Station Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(2, 3, 4)
plt.plot(history_cnn_all.history['loss'], label='Train')
plt.plot(history_cnn_all.history['val_loss'], label='Validation')
plt.title('CNN with All Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(2, 3, 5)
plt.plot(history_cnn_sat.history['loss'], label='Train')
plt.plot(history_cnn_sat.history['val_loss'], label='Validation')
plt.title('CNN with Satellite Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(2, 3, 6)
plt.plot(history_cnn_ws.history['loss'], label='Train')
plt.plot(history_cnn_ws.history['val_loss'], label='Validation')
plt.title('CNN with Weather Station Features')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

