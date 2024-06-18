import pandas as pd
import numpy as np
from utils.find_closest_lat_lon import find_closest_lat_lon
from utils.reproject_lats_lons import reproject_lats_lons
from tqdm import tqdm
from scipy.spatial import cKDTree
import argparse
import logging
import os

def find_closest_station(lat, lon, station_coords):
    tree = cKDTree(station_coords)
    distance, index = tree.query((lat, lon))
    return index

def truncate(value, decimals):
    if decimals == -1:
        return value
    factor = 10 ** decimals
    return np.floor(value * factor) / factor

def clean_sinan_dataset(sinan_df, cnes_df):
    if 'count' in sinan_df.columns:
        sinan_df.rename(columns={'count': 'CASES'}, inplace=True)

    sinan_df['acc_sat'] = np.nan
    sinan_df['avg_sat'] = np.nan
    sinan_df['max_sat'] = np.nan
    sinan_df['min_sat'] = np.nan

    sinan_df['acc_ws'] = np.nan    
    sinan_df['avg_ws'] = np.nan
    sinan_df['max_ws'] = np.nan
    sinan_df['min_ws'] = np.nan   
    
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].str.strip()
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'] != '']
    
    valid_ids = set(cnes_df['CNES'])
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'].isin(valid_ids)]
    return sinan_df

def build_dataset(sinan_path, cnes_path, lst_reference_file, rrqpe_reference_file, inmet_path, output_path):
    # Variáveis de parâmetros
    prec_significativa = 10 # mm - 
    temperatura_otima_eclosao = 25
    temperatura_otima_reproducao = 25
    umidade_otima_reproducao = 60
    limiar_chuva_intensa = 50  # mm
    limiar_chuva_extrema = 100  # mm
    limiar_risco_lixiviacao = 20  # mm
    population_size = 100000  # Placeholder para o tamanho da população, ajuste conforme necessário

    sinan_df = pd.read_parquet(sinan_path)
    cnes_df = pd.read_parquet(cnes_path)
    inmet_df = pd.read_parquet(inmet_path)
    inmet_df['VL_LATITUDE'] = inmet_df['VL_LATITUDE'].astype(float)
    inmet_df['VL_LONGITUDE'] = inmet_df['VL_LONGITUDE'].astype(float)
    inmet_df['DT_MEDICAO'] = pd.to_datetime(inmet_df['DT_MEDICAO'], format='%Y-%m-%d')

    lst_lons, lst_lats = reproject_lats_lons(lst_reference_file) 
    rrqpe_lons, rrqpe_lats = reproject_lats_lons(rrqpe_reference_file) 

    sinan_df = clean_sinan_dataset(sinan_df, cnes_df)

    cnes_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    id_unit_to_lat_lon = cnes_df.set_index('CNES')[['LAT', 'LNG']].to_dict('index')

    lst_lat_lon_cache = {}
    rrqpe_lat_lon_cache = {}

    station_coords = inmet_df[['VL_LATITUDE', 'VL_LONGITUDE']].values

    # Add columns for derived data
    derived_columns = [
        'Temp_Mov_7d_WS', 'Temp_Mov_14d_WS', 'Temp_Mov_30d_WS', 'Temp_Max_WS', 'Temp_Min_WS', 'Amp_Termica_WS',
        'Dias_Temp_Acima_20_WS', 'Anomalia_Temp_WS', 'Prec_Acum_7d_WS', 'Prec_Acum_14d_WS', 'Prec_Acum_30d_WS',
        'Dias_Prec_Significativa_WS', 'Prec_Mov_7d_WS', 'Prec_Mov_14d_WS', 'Prec_Mov_30d_WS', 'Dias_Sem_Prec_WS',
        'Indice_Ambiental_WS', 'Condicoes_Otimas_Reproducao_WS', 'Relacao_Temp_Prec_Acumulada_WS', 'Temp_Otima_Eclosao_WS',
        'Dias_Temp_Otima_WS', 'Periodo_Incubacao_Ovos_WS', 'Dias_Condicoes_Favoraveis_Eclosao_WS', 'Chuvas_Intensas_WS',
        'Dias_Prec_Extrema_WS', 'Periodos_Chuva_Continua_WS', 'Risco_Lixiviacao_WS', 'Taxa_Incidencia_Dengue_WS',
        'Casos_por_100000_Habitantes_WS', 'Risco_Infeccao_WS', 'Dias_Desde_Ultimo_Caso_WS', 'Taxa_Crescimento_Casos_WS',
        'Indice_Reproducao_Mosquito_WS', 'Dias_Condicoes_Otimas_Reproducao_WS', 'Estimativa_Populacao_Mosquitos_WS',
        'Periodos_Alta_Reprodutividade_WS', 'Num_Ciclos_Vida_Completos_WS'
    ]
    for col in derived_columns:
        sinan_df[col] = np.nan

    for index, row in tqdm(sinan_df.iterrows(), total=sinan_df.shape[0], desc="Processing rows"):
        id_unidade = row['ID_UNIDADE']
        if id_unidade not in lst_lat_lon_cache:
            lat_lon = id_unit_to_lat_lon.get(id_unidade)
            if lat_lon:
                lat, lon = lat_lon['LAT'], lat_lon['LNG']
                lst_x, lst_y = find_closest_lat_lon(lat, lon, lst_lats, lst_lons)
                rrqpe_x, rrqpe_y = find_closest_lat_lon(lat, lon, rrqpe_lats, rrqpe_lons)                
                lst_lat_lon_cache[id_unidade] = (lst_x, lst_y)
                rrqpe_lat_lon_cache[id_unidade] = (rrqpe_x, rrqpe_y)
            else:
                continue
        else:
            lst_x, lst_y = lst_lat_lon_cache[id_unidade]
            rrqpe_x, rrqpe_y = rrqpe_lat_lon_cache[id_unidade]

        dt_notific = row['DT_NOTIFIC']
        dt_notific_date = pd.to_datetime(dt_notific, format='%Y%m%d')

        lst_datasets_path = 'data/processed/lst'
        if os.path.isfile(f'{lst_datasets_path}/{dt_notific}.npz'):
            lst_data = np.load(f'{lst_datasets_path}/{dt_notific}.npz')
        else:
            continue

        rrqpe_datasets_path = 'data/processed/rrqpe'
        if os.path.isfile(f'{rrqpe_datasets_path}/{dt_notific}.npz'):
            rrqpe_data = np.load(f'{rrqpe_datasets_path}/{dt_notific}.npz')
        else:
            continue        

        sinan_df.at[index, 'avg_sat'] = truncate(lst_data['avg'][lst_x, lst_y], 2)
        sinan_df.at[index, 'max_sat'] = truncate(lst_data['max'][lst_x, lst_y], 2)
        sinan_df.at[index, 'min_sat'] = truncate(lst_data['min'][lst_x, lst_y], 2)
        sinan_df.at[index, 'acc_sat'] = truncate(rrqpe_data['acum'][rrqpe_x, rrqpe_y], 2)

        # Find the closest weather station
        station_idx = find_closest_station(float(lat), float(lon), station_coords)
        station = inmet_df.iloc[station_idx]

        # Filter the inmet_df by date
        inmet_row = inmet_df[(inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & (inmet_df['DT_MEDICAO'] == dt_notific_date)]
        if not inmet_row.empty:
            sinan_df.at[index, 'acc_ws'] = truncate(inmet_row['CHUVA'].values[0], 2)
            sinan_df.at[index, 'avg_ws'] = truncate(inmet_row['TEM_AVG'].values[0], 2)
            sinan_df.at[index, 'max_ws'] = truncate(inmet_row['TEM_MAX'].values[0], 2)
            sinan_df.at[index, 'min_ws'] = truncate(inmet_row['TEM_MIN'].values[0], 2)
            
            # Calculate moving averages for temperature
            sinan_df.at[index, 'Amp_Termica_WS'] = truncate(inmet_row['TEM_MAX'].values[0] - inmet_row['TEM_MIN'].values[0], 2)
            sinan_df.at[index, 'Temp_Mov_7d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=7))
            ]['TEM_AVG'].mean()
            sinan_df.at[index, 'Temp_Mov_14d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=14))
            ]['TEM_AVG'].mean()
            sinan_df.at[index, 'Temp_Mov_30d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30))
            ]['TEM_AVG'].mean()

            # Precipitation calculations
            sinan_df.at[index, 'Prec_Acum_7d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=7))
            ]['CHUVA'].sum()
            sinan_df.at[index, 'Prec_Acum_14d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=14))
            ]['CHUVA'].sum()
            sinan_df.at[index, 'Prec_Acum_30d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30))
            ]['CHUVA'].sum()
            
            # Count significant precipitation days
            sinan_df.at[index, 'Dias_Prec_Significativa_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['CHUVA'] >= prec_significativa)
            ].shape[0]

            sinan_df.at[index, 'Dias_Sem_Prec_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['CHUVA'] < prec_significativa) 
            ].shape[0]         
            
            # Calculate moving averages for precipitation
            sinan_df.at[index, 'Prec_Mov_7d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=7))
            ]['CHUVA'].mean()
            sinan_df.at[index, 'Prec_Mov_14d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=14))
            ]['CHUVA'].mean()
            sinan_df.at[index, 'Prec_Mov_30d_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30))
            ]['CHUVA'].mean()

            # Calculate optimal eclosion temperature and days with optimal temperature
            sinan_df.at[index, 'Temp_Otima_Eclosao_WS'] = 1 if inmet_row['TEM_AVG'].values[0] >= temperatura_otima_eclosao else 0
            sinan_df.at[index, 'Dias_Temp_Otima_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['TEM_AVG'] >= temperatura_otima_eclosao)
            ].shape[0]

            # Calculate days with favorable eclosion conditions
            sinan_df.at[index, 'Dias_Condicoes_Favoraveis_Eclosao_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) & 
                (inmet_df['TEM_AVG'] >= temperatura_otima_eclosao) & 
                (inmet_df['CHUVA'] >= 5)
            ].shape[0]

            sinan_df.at[index, 'Dias_Prec_Extrema_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) &
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30)) &
                (inmet_df['CHUVA'] >= limiar_chuva_extrema)  # Arbitrary threshold for extreme rainfall
            ].shape[0]

            sinan_df.at[index, 'Periodos_Chuva_Continua_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) &
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30)) &
                (inmet_df['CHUVA'] > 0)
            ].groupby((inmet_df['CHUVA'] > 0).ne((inmet_df['CHUVA'] > 0).shift()).cumsum()).size().max()

            sinan_df.at[index, 'Risco_Lixiviacao_WS'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) &
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30)) &
                (inmet_df['CHUVA'] > limiar_risco_lixiviacao)  # Arbitrary threshold for leaching risk
            ].shape[0]

            # Derived calculations needing additional implementation
            sinan_df.at[index, 'Anomalia_Temp_WS'] = sinan_df.at[index, 'avg_ws'] - inmet_df[(inmet_df['DT_MEDICAO'] <= dt_notific_date) & (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=30))]['TEM_AVG'].mean()
            sinan_df.at[index, 'Relacao_Temp_Prec_Acumulada_WS'] = sinan_df.at[index, 'avg_ws'] / sinan_df.at[index, 'Prec_Acum_30d_WS'] if sinan_df.at[index, 'Prec_Acum_30d_WS'] != 0 else np.nan
            sinan_df.at[index, 'Periodo_Incubacao_Ovos_WS'] = np.nan  # Placeholder para um cálculo mais complexo
            sinan_df.at[index, 'Risco_Infeccao_WS'] = sinan_df.at[index, 'Casos_por_100000_Habitantes_WS'] * sinan_df.at[index, 'Estimativa_Populacao_Mosquitos_WS'] / 100000
            sinan_df.at[index, 'Taxa_Crescimento_Casos_WS'] = (sinan_df.at[index, 'CASES'] - previous_cases['CASES'].sum()) / previous_cases['CASES'].sum() if previous_cases['CASES'].sum() != 0 else np.nan
            sinan_df.at[index, 'Dias_Condicoes_Otimas_Reproducao_WS'] = np.nan  # Placeholder para um cálculo mais complexo
            sinan_df.at[index, 'Num_Ciclos_Vida_Completos_WS'] = np.nan  # Placeholder para um cálculo mais complexo

            # Calculate additional derived columns
            sinan_df.at[index, 'Indice_Ambiental_WS'] = sinan_df.at[index, 'Temp_Mov_30d_WS'] * sinan_df.at[index, 'Prec_Acum_30d_WS']
            sinan_df.at[index, 'Estimativa_Populacao_Mosquitos_WS'] = (sinan_df.at[index, 'Temp_Mov_30d_WS'] + sinan_df.at[index, 'Prec_Acum_30d_WS']) / 2
            sinan_df.at[index, 'Indice_Reproducao_Mosquito_WS'] = sinan_df.at[index, 'Temp_Max_WS'] - sinan_df.at[index, 'Temp_Min_WS']
            sinan_df.at[index, 'Periodos_Alta_Reprodutividade_WS'] = 1 if sinan_df.at[index, 'Temp_Mov_30d_WS'] > temperatura_otima_reproducao and sinan_df.at[index, 'Prec_Acum_30d_WS'] > limiar_chuva_intensa else 0

            # Calculate days since the last case
            previous_cases = sinan_df[(sinan_df['ID_UNIDADE'] == row['ID_UNIDADE']) & (sinan_df['DT_NOTIFIC'] < dt_notific_date)]
            if not previous_cases.empty:
                last_case_date = previous_cases['DT_NOTIFIC'].max()
                sinan_df.at[index, 'Dias_Desde_Ultimo_Caso_WS'] = (dt_notific_date - last_case_date).days

            # Calculate incidence rate
            sinan_df.at[index, 'Taxa_Incidencia_Dengue_WS'] = (sinan_df.at[index, 'CASES'] / population_size) * 100000
            sinan_df.at[index, 'Casos_por_100000_Habitantes_WS'] = (sinan_df.at[index, 'CASES'] / population_size) * 100000

    # Additional cleaning steps
    # Drop the ID_AGRAVO column if it exists
    if 'ID_AGRAVO' in sinan_df.columns:
        sinan_df = sinan_df.drop(columns=['ID_AGRAVO'])           

    # Drop unnamed columns
    sinan_df = sinan_df.loc[:, ~sinan_df.columns.str.contains('^Unnamed')]

    # Convert the date column to datetime format and sort by date
    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
    sinan_df = sinan_df.dropna(subset=['DT_NOTIFIC'])  # Drop rows where date conversion failed
    sinan_df = sinan_df.sort_values('DT_NOTIFIC')

    # Check for NaN or infinite values and drop rows containing them
    sinan_df = sinan_df.replace([np.inf, -np.inf], np.nan)
    sinan_df = sinan_df.dropna()

    # Drop rows where avg_sat, max_sat, min_sat, avg_ws, max_ws, min_ws have no value
    sinan_df = sinan_df.dropna(subset=['acc_sat', 'avg_sat', 'max_sat', 'min_sat', 'acc_ws','avg_ws', 'max_ws', 'min_ws'])

    # Save the cleaned data to a new CSV file (optional)
    sinan_df.to_parquet(output_path)

def main():
    parser = argparse.ArgumentParser(description="Unify INMET datasets")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("lst_reference_file", help="Path to LST data, for reference")
    parser.add_argument("rrqpe_reference_file", help="Path to RRQPE data, for reference")
    parser.add_argument("inmet_path", help="Path to INMET data")
    parser.add_argument("output_path", help="Output path")    
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    
    build_dataset(args.sinan_path, args.cnes_path, args.lst_reference_file, args.rrqpe_reference_file, args.inmet_path, args.output_path)

if __name__ == "__main__":
    main()
