import pandas as pd
import numpy as np
from utils.find_closest_lat_lon import find_closest_lat_lon
from utils.reproject_lats_lons import reproject_lats_lons
from tqdm import tqdm

def clean_sinan_dataset(sinan_df, cnes_df):
    sinan_df['avg_sat'] = np.nan
    sinan_df['max_sat'] = np.nan
    sinan_df['min_sat'] = np.nan
    
    sinan_df['ID_UNIDADE'] = sinan_df['ID_UNIDADE'].str.strip()
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'] != '']
    
    valid_ids = set(cnes_df['CNES'])
    
    sinan_df = sinan_df[sinan_df['ID_UNIDADE'].isin(valid_ids)]
    return sinan_df

def build_dataset():
    sinan_path = 'data/processed/sinan/DENGBR23.parquet'
    cnes_path = 'data/processed/cnes/STRJ2311.parquet'
    lst_reference_file = "data/raw/lst/20230116/OR_ABI-L2-LSTF-M6_G16_s20230152100206_e20230152109514_c20230152111229.nc"
    output_path = 'data/processed/sinan/sinan.parquet'

    sinan_df = pd.read_parquet(sinan_path)
    cnes_df = pd.read_parquet(cnes_path)

    lst_lons, lst_lats = reproject_lats_lons(lst_reference_file)    

    sinan_df = clean_sinan_dataset(sinan_df, cnes_df)

    cnes_df.dropna(subset=['LAT', 'LNG'], inplace=True)

    id_unit_to_lat_lon = cnes_df.set_index('CNES')[['LAT', 'LNG']].to_dict('index')

    lst_lat_lon_cache = {}

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
        lst_datasets_path = 'data/processed/lst'
        lst_data = np.load(f'{lst_datasets_path}/{dt_notific}.npz')

        sinan_df.at[index, 'avg'] = lst_data['avg'][lst_x, lst_y].round(2)
        sinan_df.at[index, 'max'] = lst_data['max'][lst_x, lst_y].round(2)
        sinan_df.at[index, 'min'] = lst_data['min'][lst_x, lst_y].round(2)

    sinan_df.to_parquet(output_path)

def main():
    build_dataset()

if __name__ == "__main__":
    main()    
