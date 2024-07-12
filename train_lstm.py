import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import shap
import argparse

# Função para construir, treinar e visualizar modelos LSTM
def train_and_visualize_lstm(X_train, y_train, X_test, y_test, unit_id, features, model_name, output_path, performance_metrics):
    # Escalar os dados
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Redimensionar os dados para LSTM [samples, timesteps, features]
    timesteps = 1
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, X_test_scaled.shape[1]))

    # Construir o modelo LSTM
    model_lstm = Sequential([
        Input(shape=(timesteps, X_train_scaled.shape[1])),
        LSTM(50),
        Dense(1)
    ])
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Treinar o modelo LSTM
    history_lstm = model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test))

    # Previsões e avaliação do modelo LSTM
    y_pred_lstm = model_lstm.predict(X_test_lstm)
    y_pred_lstm_rounded = np.round(y_pred_lstm).astype(int)
    lstm_loss = mean_squared_error(y_test, y_pred_lstm_rounded)
    logging.info(f'LSTM Test MSE for {unit_id} ({model_name}): {lstm_loss}')

    # Calcular métricas adicionais
    total_cases_train = y_train.sum()
    estimated_cases_test = y_pred_lstm_rounded.sum()
    actual_cases_test = y_test.sum()

    # Armazenar a métrica de desempenho
    performance_metrics.loc[len(performance_metrics)] = [unit_id, model_name, lstm_loss, total_cases_train, estimated_cases_test, actual_cases_test]

    # Verificar Overfitting
    train_loss_lstm = history_lstm.history['loss']
    val_loss_lstm = history_lstm.history['val_loss']
    plt.figure(figsize=(14, 7))
    plt.plot(train_loss_lstm, label='Train Loss')
    plt.plot(val_loss_lstm, label='Validation Loss')
    plt.title(f'LSTM Train vs Validation Loss for ID_UNIDADE: {unit_id} ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, f'lstm_train_val_loss_{unit_id}_{model_name}.png'))
    plt.close()

    # Visualizar resultados LSTM
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred_lstm_rounded, label='Predicted')
    plt.title(f'LSTM Predictions vs Actuals for ID_UNIDADE: {unit_id} ({model_name})')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.legend()
    plt.savefig(os.path.join(output_path, f'lstm_results_{unit_id}_{model_name}.png'))
    plt.close()

    # Calcular a importância das features usando SHAP com KernelExplainer
    # def f_lstm(x):
    #     return model_lstm.predict(x.reshape((x.shape[0], timesteps, x.shape[1])))

    # explainer_lstm = shap.KernelExplainer(f_lstm, X_train_scaled)
    # shap_values_lstm = explainer_lstm.shap_values(X_test_scaled)
    # shap.summary_plot(shap_values_lstm, X_test_scaled, feature_names=features, show=False)
    # plt.savefig(os.path.join(output_path, f'shap_summary_lstm_{unit_id}_{model_name}.png'))
    # plt.close()

def train(dataset_path, output_path, split_date, id_unidade):
    # Carregar o dataset
    df = pd.read_parquet(dataset_path)

    # Filtrar pelo ID_UNIDADE se fornecido
    if id_unidade:
        df = df[df["ID_UNIDADE"] == id_unidade]

    df.to_csv("teste.csv")        

    # Selecionar as features e a variável alvo
    features_cases = [
        'CASES', 'CASES_MM_14', 'CASES_MM_21', 'CASES_ACC_14', 'CASES_ACC_21'
    ]

    features_inmet = [
        'IDEAL_TEMP_INMET', 'EXTREME_TEMP_INMET', 'SIGNIFICANT_RAIN_INMET', 'EXTREME_RAIN_INMET',
        'TEMP_RANGE_INMET',
        'TEM_AVG_INMET_MM_7', 'TEM_AVG_INMET_MM_14', 'TEM_AVG_INMET_MM_21',
        'CHUVA_INMET_MM_7', 'CHUVA_INMET_MM_14', 'CHUVA_INMET_MM_21',
        'TEMP_RANGE_INMET_MM_7', 'TEMP_RANGE_INMET_MM_14', 'TEMP_RANGE_INMET_MM_21',
        'TEM_AVG_INMET_ACC_7', 'TEM_AVG_INMET_ACC_14', 'TEM_AVG_INMET_ACC_21',
        'CHUVA_INMET_ACC_7', 'CHUVA_INMET_ACC_14', 'CHUVA_INMET_ACC_21'
    ]

    features_sat = [
        'IDEAL_TEMP_SAT', 'EXTREME_TEMP_SAT', 'SIGNIFICANT_RAIN_SAT', 'EXTREME_RAIN_SAT',
        'TEMP_RANGE_SAT',
        'TEM_AVG_SAT_MM_7', 'TEM_AVG_SAT_MM_14', 'TEM_AVG_SAT_MM_21',
        'CHUVA_SAT_MM_7', 'CHUVA_SAT_MM_14', 'CHUVA_SAT_MM_21',
        'TEMP_RANGE_SAT_MM_7', 'TEMP_RANGE_SAT_MM_14', 'TEMP_RANGE_SAT_MM_21',
        'TEM_AVG_SAT_ACC_7', 'TEM_AVG_SAT_ACC_14', 'TEM_AVG_SAT_ACC_21',
        'CHUVA_SAT_ACC_7', 'CHUVA_SAT_ACC_14', 'CHUVA_SAT_ACC_21'
    ]

    all_features = features_cases + features_inmet + features_sat
    inmet_and_cases = features_cases + features_sat
    sat_and_cases = features_sat + features_sat
    target = 'CASES'

    # Remover NaNs
    df = df.dropna()

    # Agrupar por ID_UNIDADE
    grouped = df.groupby('ID_UNIDADE')

    # DataFrame para armazenar as métricas de desempenho
    performance_metrics = pd.DataFrame(columns=['ID_UNIDADE', 'Model', 'MSE', 'Total_Cases_Train', 'Estimated_Cases_Test', 'Actual_Cases_Test'])

    # Iterar sobre cada grupo de ID_UNIDADE
    for name, group in grouped:
        logging.info(f'Treinando modelos LSTM para ID_UNIDADE: {name}')
        group = group.sort_values(by='DT_NOTIFIC')
        
        # Dividir os dados em conjuntos de treino e teste
        train_df = group[group['DT_NOTIFIC'] <= split_date]
        test_df = group[group['DT_NOTIFIC'] > split_date]

        X_train_all_features = train_df[all_features].values
        y_train = train_df[target].values
        X_test_all_features = test_df[all_features].values
        y_test = test_df[target].values

        # Treinar e avaliar os modelos LSTM com todas as features
        train_and_visualize_lstm(X_train_all_features, y_train, X_test_all_features, y_test, name, all_features, "all_features", output_path, performance_metrics)

        # Treinar e avaliar os modelos LSTM sem features climáticas
        X_train_inmet_and_cases = train_df[inmet_and_cases].values
        X_test_inmet_and_cases = test_df[inmet_and_cases].values
        train_and_visualize_lstm(X_train_inmet_and_cases, y_train, X_test_inmet_and_cases, y_test, name, inmet_and_cases, "inmet_features", output_path, performance_metrics)

        X_train_sat_and_cases = train_df[sat_and_cases].values
        X_test_sat_and_cases = test_df[sat_and_cases].values
        train_and_visualize_lstm(X_train_sat_and_cases, y_train, X_test_sat_and_cases, y_test, name, sat_and_cases, "sat_features", output_path, performance_metrics)                

        features_without_climate = features_cases
        X_train_no_climate = train_df[features_without_climate].values
        X_test_no_climate = test_df[features_without_climate].values
        train_and_visualize_lstm(X_train_no_climate, y_train, X_test_no_climate, y_test, name, features_without_climate, "no_climate_features", output_path, performance_metrics)

    train_df = df[df['DT_NOTIFIC'] <= split_date]
    test_df = df[df['DT_NOTIFIC'] > split_date]        

    X_train_all_features = train_df[all_features].values
    y_train = train_df[target].values
    X_test_all_features = test_df[all_features].values
    y_test = test_df[target].values

    # Treinar e avaliar os modelos LSTM com todas as features
    train_and_visualize_lstm(X_train_all_features, y_train, X_test_all_features, y_test, "all", all_features, "all_features", output_path, performance_metrics)

    # Treinar e avaliar os modelos LSTM sem features climáticas
    X_train_inmet_and_cases = train_df[inmet_and_cases].values
    X_test_inmet_and_cases = test_df[inmet_and_cases].values
    train_and_visualize_lstm(X_train_inmet_and_cases, y_train, X_test_inmet_and_cases, y_test, "all", inmet_and_cases, "inmet_features", output_path, performance_metrics)

    X_train_sat_and_cases = train_df[sat_and_cases].values
    X_test_sat_and_cases = test_df[sat_and_cases].values
    train_and_visualize_lstm(X_train_sat_and_cases, y_train, X_test_sat_and_cases, y_test, "all", sat_and_cases, "sat_features", output_path, performance_metrics)                

    features_without_climate = features_cases
    X_train_no_climate = train_df[features_without_climate].values
    X_test_no_climate = test_df[features_without_climate].values
    train_and_visualize_lstm(X_train_no_climate, y_train, X_test_no_climate, y_test, "all", features_without_climate, "no_climate_features", output_path, performance_metrics)    

    # Salvar métricas de desempenho em um arquivo CSV
    performance_metrics.to_csv(os.path.join(output_path, 'lstm_performance_metrics.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description="Train LSTM models for dengue case prediction")
    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument("output_path", help="Path to the output")
    parser.add_argument("split_date", help="Date to split test/train (YYYY-MM-DD)")
    parser.add_argument("--id_unidade", dest="id_unidade", default=None, help="Filter an ID_UNIDADE")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    train(args.dataset_path, args.output_path, args.split_date, args.id_unidade)

if __name__ == '__main__':
    main()
    #train(
    #    dataset_path="data/processed/sinan/sinan.parquet",
    #    output_path="data/processed/lstm",
    #    split_date="2022-12-31",
    #    id_unidade="2296306",
    #)