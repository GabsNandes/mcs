import pandas as pd
import numpy as np
from utils.find_closest_lat_lon import find_closest_lat_lon
from utils.reproject_lats_lons import reproject_lats_lons
from tqdm import tqdm
from scipy.spatial import cKDTree
import argparse
import logging
import os
from joblib import Parallel, delayed

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

    derived_columns_ws = [
        'temp_mov_7d_ws', 'temp_mov_14d_ws', 'temp_mov_30d_ws', 
        'prec_mov_7d_ws', 'prec_mov_14d_ws', 'prec_mov_30d_ws', 
        'prec_acum_7d_ws', 'prec_acum_14d_ws', 'prec_acum_30d_ws',
        'amp_termica_ws', 'temp_ideal_ws','temp_extrema_ws', 
        'prec_significativa_ws', 'prec_extrema_ws',
    ]

    derived_columns_sat = [
        'temp_mov_7d_sat', 'temp_mov_14d_sat', 'temp_mov_30d_sat', 
        'prec_mov_7d_sat', 'prec_mov_14d_sat', 'prec_mov_30d_sat', 
        'prec_acum_7d_sat', 'prec_acum_14d_sat', 'prec_acum_30d_sat',
        'amp_termica_sat', 'temp_ideal_sat','temp_extrema_sat', 
        'prec_significativa_sat', 'prec_extrema_sat',
    ]    

    for col in derived_columns_ws:
        sinan_df[col] = np.nan

    for col in derived_columns_sat:
        sinan_df[col] = np.nan          
    
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].str.strip()
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'] != '']
    
    valid_ids = set(cnes_df['CNES'])
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'].isin(valid_ids)]
    return sinan_df

def process_row(index, row, id_unit_to_lat_lon, lst_lat_lon_cache, rrqpe_lat_lon_cache, lst_data_cache, rrqpe_data_cache, inmet_df, station_coords, lst_lats, lst_lons, rrqpe_lats, rrqpe_lons):
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
            return None
    else:
        lst_x, lst_y = lst_lat_lon_cache[id_unidade]
        rrqpe_x, rrqpe_y = rrqpe_lat_lon_cache[id_unidade]

    dt_notific = row['DT_NOTIFIC']
    dt_notific_date = pd.to_datetime(dt_notific, format='%Y%m%d')

    lst_datasets_path = 'data/processed/lst'
    if os.path.isfile(f'{lst_datasets_path}/{dt_notific}.npz'):
        if dt_notific not in lst_data_cache:
            lst_data_cache[dt_notific] = np.load(f'{lst_datasets_path}/{dt_notific}.npz')
        lst_data = lst_data_cache[dt_notific]
    else:
        return None

    rrqpe_datasets_path = 'data/processed/rrqpe'
    if os.path.isfile(f'{rrqpe_datasets_path}/{dt_notific}.npz'):
        if dt_notific not in rrqpe_data_cache:
            rrqpe_data_cache[dt_notific] = np.load(f'{rrqpe_datasets_path}/{dt_notific}.npz')
        rrqpe_data = rrqpe_data_cache[dt_notific]
    else:
        return None        

    result = {
        'index': index,
        'avg_sat': truncate(lst_data['avg'][lst_x, lst_y], 2),
        'max_sat': truncate(lst_data['max'][lst_x, lst_y], 2),
        'min_sat': truncate(lst_data['min'][lst_x, lst_y], 2),
        'acc_sat': truncate(rrqpe_data['acum'][rrqpe_x, rrqpe_y], 2),
        'temp_mov_7d_sat': np.nan, 'temp_mov_14d_sat': np.nan, 'temp_mov_30d_sat': np.nan,
        'prec_mov_7d_sat': np.nan, 'prec_mov_14d_sat': np.nan, 'prec_mov_30d_sat': np.nan,
        'prec_acum_7d_sat': np.nan, 'prec_acum_14d_sat': np.nan, 'prec_acum_30d_sat': np.nan,
        'amp_termica_sat': np.nan, 'temp_ideal_sat': np.nan, 'temp_extrema_sat': np.nan,
        'prec_significativa_sat': np.nan, 'prec_extrema_sat': np.nan,
        'acc_ws': np.nan, 'avg_ws': np.nan, 'max_ws': np.nan, 'min_ws': np.nan,
        'temp_mov_7d_ws': np.nan, 'temp_mov_14d_ws': np.nan, 'temp_mov_30d_ws': np.nan,
        'prec_mov_7d_ws': np.nan, 'prec_mov_14d_ws': np.nan, 'prec_mov_30d_ws': np.nan,
        'prec_acum_7d_ws': np.nan, 'prec_acum_14d_ws': np.nan, 'prec_acum_30d_ws': np.nan,
        'amp_termica_ws': np.nan, 'temp_ideal_ws': np.nan, 'temp_extrema_ws': np.nan,
        'prec_significativa_ws': np.nan, 'prec_extrema_ws': np.nan
    }

    temp_sum = {7: [], 14: [], 30: []}
    prec_sum = {7: [], 14: [], 30: []}
    
    for days in [7, 14, 30]:
        for delta in range(days):
            current_date = dt_notific_date - pd.Timedelta(days=delta)
            current_date_str = current_date.strftime('%Y%m%d')
            
            if os.path.isfile(f'{lst_datasets_path}/{current_date_str}.npz'):
                if current_date_str not in lst_data_cache:
                    lst_data_cache[current_date_str] = np.load(f'{lst_datasets_path}/{current_date_str}.npz')
                temp_sum[days].append(lst_data_cache[current_date_str]['avg'][lst_x, lst_y])
            
            if os.path.isfile(f'{rrqpe_datasets_path}/{current_date_str}.npz'):
                if current_date_str not in rrqpe_data_cache:
                    rrqpe_data_cache[current_date_str] = np.load(f'{rrqpe_datasets_path}/{current_date_str}.npz')
                prec_sum[days].append(rrqpe_data_cache[current_date_str]['acum'][rrqpe_x, rrqpe_y])

        if temp_sum[days]:
            result[f'temp_mov_{days}d_sat'] = truncate(np.mean(temp_sum[days]), 2)
        if prec_sum[days]:
            result[f'prec_mov_{days}d_sat'] = truncate(np.mean(prec_sum[days]), 2)
            result[f'prec_acum_{days}d_sat'] = truncate(np.sum(prec_sum[days]), 2)
    
    result['amp_termica_sat'] = truncate(result['max_sat'] - result['min_sat'], 2)
    result['temp_ideal_sat'] = 1 if 21 <= result['avg_sat'] <= 27 else 0
    result['temp_extrema_sat'] = 1 if result['avg_sat'] <= 14 or result['avg_sat'] >= 38 else 0
    result['prec_significativa_sat'] = 1 if 10 <= result['acc_sat'] <= 150 else 0
    result['prec_extrema_sat'] = 1 if result['acc_sat'] >= 150 else 0

    station_idx = find_closest_station(float(lat), float(lon), station_coords)
    station = inmet_df.iloc[station_idx]

    inmet_row = inmet_df[(inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & (inmet_df['DT_MEDICAO'] == dt_notific_date)]
    if not inmet_row.empty:
        result['acc_ws'] = truncate(inmet_row['CHUVA'].values[0], 2)
        result['avg_ws'] = truncate(inmet_row['TEM_AVG'].values[0], 2)
        result['max_ws'] = truncate(inmet_row['TEM_MAX'].values[0], 2)
        result['min_ws'] = truncate(inmet_row['TEM_MIN'].values[0], 2)
        for days in [7, 14, 30]:
            result[f'temp_mov_{days}d_ws'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=days))
            ]['TEM_AVG'].mean()
            result[f'prec_mov_{days}d_ws'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=days))
            ]['CHUVA'].mean()
            result[f'prec_acum_{days}d_ws'] = inmet_df[
                (inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & 
                (inmet_df['DT_MEDICAO'] <= dt_notific_date) &
                (inmet_df['DT_MEDICAO'] >= dt_notific_date - pd.Timedelta(days=days))
            ]['CHUVA'].sum()

        result['amp_termica_ws'] = truncate(inmet_row['TEM_MAX'].values[0] - inmet_row['TEM_MIN'].values[0], 2)
        result['temp_ideal_ws'] = 1 if 21 <= inmet_row['TEM_AVG'].values[0] <= 27 else 0
        result['temp_extrema_ws'] = 1 if inmet_row['TEM_AVG'].values[0] <= 14 or inmet_row['TEM_AVG'].values[0] >= 38 else 0
        result['prec_significativa_ws'] = 1 if 10 <= inmet_row['CHUVA'].values[0] <= 150 else 0
        result['prec_extrema_ws'] = 1 if inmet_row['CHUVA'].values[0] >= 150 else 0

    return result

def build_dataset(sinan_path, cnes_path, lst_reference_file, rrqpe_reference_file, inmet_path, output_path):
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
    lst_data_cache = {}
    rrqpe_data_cache = {}
    station_coords = inmet_df[['VL_LATITUDE', 'VL_LONGITUDE']].values

    results = Parallel(n_jobs=-1)(delayed(process_row)(
        index, row, id_unit_to_lat_lon, lst_lat_lon_cache, rrqpe_lat_lon_cache, lst_data_cache, rrqpe_data_cache, inmet_df, station_coords, lst_lats, lst_lons, rrqpe_lats, rrqpe_lons
    ) for index, row in tqdm(sinan_df.iterrows(), total=sinan_df.shape[0], desc="Processing rows"))

    for result in results:
        if result:
            for key, value in result.items():
                if key != 'index':
                    sinan_df.at[result['index'], key] = value

    # Additional cleaning steps
    if 'ID_AGRAVO' in sinan_df.columns:
        sinan_df = sinan_df.drop(columns=['ID_AGRAVO'])           

    sinan_df = sinan_df.loc[:, ~sinan_df.columns.str.contains('^Unnamed')]

    sinan_df['DT_NOTIFIC'] = pd.to_datetime(sinan_df['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
    sinan_df = sinan_df.dropna(subset=['DT_NOTIFIC'])
    sinan_df = sinan_df.sort_values('DT_NOTIFIC')

    sinan_df = sinan_df.replace([np.inf, -np.inf], np.nan)
    sinan_df = sinan_df.dropna(subset=['acc_sat', 'avg_sat', 'max_sat', 'min_sat', 'acc_ws','avg_ws', 'max_ws', 'min_ws'])

    sinan_df.to_parquet(output_path)
    sinan_df.to_csv('teste.csv')

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
