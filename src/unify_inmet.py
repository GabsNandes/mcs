import logging
import pandas as pd
import argparse
import os
from utils.reproject_lats_lons import reproject_lats_lons
from utils.find_closest_lat_lon import find_closest_lat_lon

def unify_inmet(raw_lst_path, raw_inmet_path, processed_inmet_path, aggregated):
    columns = ['CD_ESTACAO', 'DT_MEDICAO', 'HR_MEDICAO', 'TEM_MIN', 'TEM_MAX', 'TEM_INS', 'VL_LATITUDE', 'VL_LONGITUDE']
    dfs = []
    os.makedirs(processed_inmet_path, exist_ok=True)

    nc_files = os.listdir(raw_lst_path)
    lst_reference_file = os.path.join(raw_lst_path, nc_files[0])

    lons, lats = reproject_lats_lons(lst_reference_file)    
    
    for filename in os.listdir(raw_inmet_path):
        logging.info(f"Adding: {filename} to dataset")
        if filename.endswith('.parquet'):
            file_path = os.path.join(raw_inmet_path, filename)
            df = pd.read_parquet(file_path)           
            lat = df['VL_LATITUDE'][0]
            lon = df['VL_LONGITUDE'][0]
            x, y = find_closest_lat_lon(lat, lon, lats, lons)           
            df = df[columns]
            df['x'] = x
            df['y'] = y
            
            df['TEM_MIN'] = pd.to_numeric(df['TEM_MIN'], errors='coerce')
            df['TEM_MAX'] = pd.to_numeric(df['TEM_MAX'], errors='coerce')
            df['TEM_INS'] = pd.to_numeric(df['TEM_INS'], errors='coerce')
            
            dfs.append(df)
   
    df_concat = pd.concat(dfs, ignore_index=True)
    
    if aggregated:
        df_aggregated = df_concat.groupby(['CD_ESTACAO', 'DT_MEDICAO']).agg({
            'TEM_MIN': 'min',
            'TEM_MAX': 'max',
            'TEM_INS': 'mean',
            'VL_LATITUDE': 'first',
            'VL_LONGITUDE': 'first'            
        }).reset_index()
        df_aggregated.columns = ['CD_ESTACAO', 'DT_MEDICAO', 'TEM_MIN', 'TEM_MAX', 'TEM_AVG', 'VL_LATITUDE', 'VL_LONGITUDE']

        logging.info(f"Saved aggregated inmet file at: {processed_inmet_path}/aggregated.parquet")

        df_aggregated.to_parquet(os.path.join(processed_inmet_path, 'aggregated.parquet'))
    else:
        df_concat.to_parquet(os.path.join(processed_inmet_path, 'concat.parquet'))

        logging.info(f"Saved concatenated inmet file at: {processed_inmet_path}/concat.parquet")

def main():
    parser = argparse.ArgumentParser(description="Unify INMET datasets")
    parser.add_argument("raw_lst_path", help="Path to LST data, for reference")
    parser.add_argument("raw_inmet_path", help="Path to INMET data")
    parser.add_argument("processed_inmet_path", help="Output path")
    parser.add_argument("--aggregated", dest="aggregated", type=bool, nargs='?', const=True, default=False, help="Set if the output will be aggregated")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    unify_inmet(args.raw_lst_path, args.raw_inmet_path, args.processed_inmet_path, args.aggregated)

if __name__ == "__main__":
    main()