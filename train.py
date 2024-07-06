import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# Carregar o dataset
df = pd.read_parquet('data/processed/sinan/sinan.parquet')

df = df[df["ID_UNIDADE"] == "2296306"]

# Selecionar as features e a variável alvo
features = [
    'IDEAL_TEMP_INMET', 'EXTREME_TEMP_INMET', 'SIGNIFICANT_RAIN_INMET', 'EXTREME_RAIN_INMET',
    'IDEAL_TEMP_SAT', 'EXTREME_TEMP_SAT', 'SIGNIFICANT_RAIN_SAT', 'EXTREME_RAIN_SAT',
    'TEMP_RANGE_INMET', 'TEMP_RANGE_SAT', 'CASES',
    'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21',
    'TEM_AVG_INMET_MM_7', 'TEM_AVG_INMET_MM_14', 'TEM_AVG_INMET_MM_21',
    'CHUVA_INMET_MM_7', 'CHUVA_INMET_MM_14', 'CHUVA_INMET_MM_21',
    'TEMP_RANGE_INMET_MM_7', 'TEMP_RANGE_INMET_MM_14', 'TEMP_RANGE_INMET_MM_21',
    'TEM_AVG_SAT_MM_7', 'TEM_AVG_SAT_MM_14', 'TEM_AVG_SAT_MM_21',
    'CHUVA_SAT_MM_7', 'CHUVA_SAT_MM_14', 'CHUVA_SAT_MM_21',
    'TEMP_RANGE_SAT_MM_7', 'TEMP_RANGE_SAT_MM_14', 'TEMP_RANGE_SAT_MM_21',
    'TEM_AVG_INMET_ACC_7', 'TEM_AVG_INMET_ACC_14', 'TEM_AVG_INMET_ACC_21',
    'CHUVA_INMET_ACC_7', 'CHUVA_INMET_ACC_14', 'CHUVA_INMET_ACC_21',
    'TEM_AVG_SAT_ACC_7', 'TEM_AVG_SAT_ACC_14', 'TEM_AVG_SAT_ACC_21',
    'CHUVA_SAT_ACC_7', 'CHUVA_SAT_ACC_14', 'CHUVA_SAT_ACC_21'
]
target = 'CASES'

# Remover NaNs
df = df.dropna()

# Agrupar por ID_UNIDADE
grouped = df.groupby('ID_UNIDADE')

# Definir a data de corte para a divisão entre treino e teste
split_date = '2022-12-31'

# Função para construir e treinar modelos
def train_models(X_train, y_train, X_test, y_test):
    # Escalar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Redimensionar os dados para LSTM [samples, timesteps, features]
    timesteps = 1
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))

    # Construir o modelo LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(timesteps, X_train_scaled.shape[1])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Treinar o modelo LSTM
    model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test))

    # Avaliar o modelo LSTM
    lstm_loss = model_lstm.evaluate(X_test_lstm, y_test)
    print(f'LSTM Test Loss: {lstm_loss}')

    # Redimensionar os dados para CNN [samples, timesteps, features, channels]
    '''
    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1], 1))
    X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1], 1))

    # Construir o modelo CNN
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, X_train_scaled.shape[1], 1)))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(50, activation='relu'))
    model_cnn.add(Dense(1))
    model_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Treinar o modelo CNN
    model_cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))

    # Avaliar o modelo CNN
    cnn_loss = model_cnn.evaluate(X_test_cnn, y_test)
    print(f'CNN Test Loss: {cnn_loss}')    
    '''

# Iterar sobre cada grupo de ID_UNIDADE
for name, group in grouped:
    print(f'Treinando modelos para ID_UNIDADE: {name}')
    group = group.sort_values(by='DT_NOTIFIC')
    
    # Dividir os dados em conjuntos de treino e teste
    train_df = group[group['DT_NOTIFIC'] <= split_date]
    test_df = group[group['DT_NOTIFIC'] > split_date]

    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values

    # Treinar e avaliar os modelos
    train_models(X_train, y_train, X_test, y_test)
