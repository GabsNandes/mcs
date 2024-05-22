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
    factor = 10 ** decimals
    return np.floor(value * factor) / factor

def clean_sinan_dataset(sinan_df, cnes_df):
    if 'count' in sinan_df.columns:
        sinan_df.rename(columns={'count': 'CASES'}, inplace=True)

    sinan_df['avg_sat'] = np.nan
    sinan_df['max_sat'] = np.nan
    sinan_df['min_sat'] = np.nan

    sinan_df['avg_ws'] = np.nan
    sinan_df['max_ws'] = np.nan
    sinan_df['min_ws'] = np.nan   

    sinan_df["lat"] = ''
    sinan_df["lon"] = ''
    
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].str.strip()
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'] != '']
    
    valid_ids = set(cnes_df['CNES'])
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'].isin(valid_ids)]
    return sinan_df

def build_dataset(sinan_path, cnes_path, lst_reference_file, inmet_path, output_path):
    sinan_df = pd.read_parquet(sinan_path)
    cnes_df = pd.read_parquet(cnes_path)
    inmet_df = pd.read_parquet(inmet_path)
    inmet_df['VL_LATITUDE'] = inmet_df['VL_LATITUDE'].astype(float)
    inmet_df['VL_LONGITUDE'] = inmet_df['VL_LONGITUDE'].astype(float)
    inmet_df['DT_MEDICAO'] = pd.to_datetime(inmet_df['DT_MEDICAO'], format='%Y-%m-%d')

    lst_lons, lst_lats = reproject_lats_lons(lst_reference_file)    

    sinan_df = clean_sinan_dataset(sinan_df, cnes_df)

    cnes_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    id_unit_to_lat_lon = cnes_df.set_index('CNES')[['LAT', 'LNG']].to_dict('index')

    lst_lat_lon_cache = {}

    station_coords = inmet_df[['VL_LATITUDE', 'VL_LONGITUDE']].values

    for index, row in tqdm(sinan_df.iterrows(), total=sinan_df.shape[0], desc="Processing rows"):
        id_unidade = row['ID_UNIDADE']
        if id_unidade not in lst_lat_lon_cache:
            lat_lon = id_unit_to_lat_lon.get(id_unidade)
            if lat_lon:
                lat, lon = lat_lon['LAT'], lat_lon['LNG']
                lst_x, lst_y = find_closest_lat_lon(lat, lon, lst_lats, lst_lons)
                lst_lat_lon_cache[id_unidade] = (lst_x, lst_y)
            else:
                continue
        else:
            lst_x, lst_y = lst_lat_lon_cache[id_unidade]

        dt_notific = row['DT_NOTIFIC']
        dt_notific_date = pd.to_datetime(dt_notific, format='%Y%m%d')

        lst_datasets_path = 'data/processed/lst'
        if os.path.isfile(f'{lst_datasets_path}/{dt_notific}.npz'):
            lst_data = np.load(f'{lst_datasets_path}/{dt_notific}.npz')
        else:
            continue

        sinan_df.at[index, 'avg_sat'] = truncate(lst_data['avg'][lst_x, lst_y], 2)
        sinan_df.at[index, 'max_sat'] = truncate(lst_data['max'][lst_x, lst_y], 2)
        sinan_df.at[index, 'min_sat'] = truncate(lst_data['min'][lst_x, lst_y], 2)
        sinan_df.at[index, 'lon'] = lat
        sinan_df.at[index, 'lat'] = lon

        # Find the closest weather station
        station_idx = find_closest_station(float(lat), float(lon), station_coords)
        station = inmet_df.iloc[station_idx]

        # Filter the inmet_df by date
        inmet_row = inmet_df[(inmet_df['CD_ESTACAO'] == station['CD_ESTACAO']) & (inmet_df['DT_MEDICAO'] == dt_notific_date)]
        if not inmet_row.empty:
            sinan_df.at[index, 'avg_ws'] = truncate(inmet_row['TEM_AVG'].values[0], 2)
            sinan_df.at[index, 'max_ws'] = truncate(inmet_row['TEM_MAX'].values[0], 2)
            sinan_df.at[index, 'min_ws'] = truncate(inmet_row['TEM_MIN'].values[0], 2)
    
    sinan_df = sinan_df.loc[:, ~sinan_df.columns.str.contains('^Unnamed')]
    sinan_df.to_parquet(output_path)
    sinan_df.head(100).to_csv('teste.csv')

def main():
    parser = argparse.ArgumentParser(description="Unify INMET datasets")
    parser.add_argument("sinan_path", help="Path to SINAN data")
    parser.add_argument("cnes_path", help="Path to CNES data")
    parser.add_argument("lst_reference_file", help="Path to LST data, for reference")
    parser.add_argument("inmet_path", help="Path to INMET data")
    parser.add_argument("output_path", help="Output path")    
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    
    build_dataset(args.sinan_path, args.cnes_path, args.lst_reference_file, args.inmet_path, args.output_path)

if __name__ == "__main__":
    main()
