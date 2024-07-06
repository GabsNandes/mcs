import pandas as pd
from utils.download_rrqpe_data import download_rrqpe_data
import logging
import os
from datetime import datetime
from datetime import timedelta
import argparse
from utils.reproject_lats_lons import reproject_lats_lons
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def download_rrqpe(path, date):
    download_rrqpe_data(date, f"{path}/{date}")

def process_file(file, date, lon, lat):
    extent = [-44.7930, -40.7635, -23.3702, -20.7634] # TODO: Move to args
    loni, lonf, lati, latf = extent
    ds = xr.open_dataset(file)
    
    data_array = ds['RRQPE']
       
    df = data_array.to_dataframe().reset_index()
    
    df['latitude'] = lat
    df['longitude'] = lon
    
    df = df[(df['latitude'] >= lati) & (df['latitude'] <= latf) & (df['longitude'] >= loni) & (df['longitude'] <= lonf)]

    df = df.dropna(subset=['RRQPE'])
    
    df['date'] = date  # Add the date column
   
    return df[['date', 'RRQPE', 'latitude', 'longitude']]

def create_dataset_from_files(path, date):
    rrqpepath = f'{path}/{date}'
    files = [os.path.join(rrqpepath, f) for f in os.listdir(rrqpepath) if f.endswith('.nc')]

    data_list = []
    lon, lat = reproject_lats_lons('data/raw/rrqpe/ref.nc')
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, date, lon=lon_flat, lat=lat_flat): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                data_list.append(future.result())
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")

    if data_list:
        combined_df = pd.concat(data_list, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=['date', 'RRQPE', 'latitude', 'longitude'])

    return combined_df

def calculate_sum_rrqpe(date, rrqpepath):
    #download_rrqpe(rrqpepath, date)
    dataset = create_dataset_from_files(rrqpepath, date)
    aggregated_df = dataset.groupby(['latitude', 'longitude']).agg(
        RRQPE_SUM=('RRQPE', 'sum'),
    ).reset_index()
    aggregated_df['date'] = date
    aggregated_list.append(aggregated_df)

aggregated_list = []
def main():   
    parser = argparse.ArgumentParser(description="Calculate RRQPE data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("rrqpepath", help="Path where rrqpe files will be downloaded, default is data/raw", default="data/raw")    
    parser.add_argument("destpath", help="Destination path, default is data/processed", default="data/processed")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating RRQPE for {current_date.strftime('%Y%m%d')}")

        calculate_sum_rrqpe(current_date.strftime('%Y%m%d'), args.rrqpepath)    
        current_date += timedelta(days=1)
    
    final_df = pd.concat(aggregated_list, ignore_index=True)
    final_df.to_parquet(args.destpath, index=False)

if __name__ == "__main__":
    main()
