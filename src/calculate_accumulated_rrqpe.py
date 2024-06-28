import pandas as pd
import logging
import os
import psutil
from datetime import datetime, timedelta
import argparse
from utils.download_rrqpe_data import download_rrqpe_data
from utils.reproject_lats_lons import reproject_lats_lons
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def download_rrqpe(path, date):
    download_rrqpe_data(date, f"{path}/{date}")

def process_file(file_info, extent):
    file, date, lon, lat = file_info
    loni, lonf, lati, latf = extent
    ds = xr.open_dataset(file)
    
    data_array = ds['RRQPE']
    
    df = data_array.to_dataframe().reset_index()
    
    df['latitude'] = lat.flatten()
    df['longitude'] = lon.flatten()
    
    df = df[(df['latitude'] >= lati) & (df['latitude'] <= latf) & (df['longitude'] >= loni) & (df['longitude'] <= lonf)]

    df = df.dropna(subset=['RRQPE'])
    
    df['date'] = date  # Add the date column
   
    return df[['date', 'RRQPE', 'latitude', 'longitude']]

def gather_files(path, dates):
    file_list = []
    for date in dates:
        date_path = os.path.join(path, date)
        files = [os.path.join(date_path, f) for f in os.listdir(date_path) if f.endswith('.nc')]
        for file in files:
            file_list.append((file, date))
    return file_list

def create_dataset_from_files(file_list, extent, max_workers):
    lon, lat = reproject_lats_lons('data/raw/rrqpe/ref.nc')

    data_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, (file, date, lon, lat), extent): (file, date) for file, date in file_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                data_list.append(future.result())
            except Exception as e:
                file, date = futures[future]
                print(f"Error processing file {file} for date {date}: {e}")

    if data_list:
        combined_df = pd.concat(data_list, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=['date', 'RRQPE', 'latitude', 'longitude'])

    return combined_df

def calculate_acum_rrqpe(file_list, extent, max_workers):
    dataset = create_dataset_from_files(file_list, extent, max_workers)
    aggregated_list.append(dataset)

aggregated_list = []

def main():   
    parser = argparse.ArgumentParser(description="Calculate RRQPE data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("rrqpepath", help="Path where rrqpe files will be downloaded, default is data/raw", default="data/raw")    
    parser.add_argument("destpath", help="Destination path, default is data/raw", default="data/raw")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    parser.add_argument("--extent", nargs=4, type=float, default=[-44.7930, -40.7635, -23.3702, -20.7634],
                        help="Bounding box for data filtering: loni lonf lati latf")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker processes to use")
    parser.add_argument("--download", type=bool, default=False, help="Download the data if set to True, skip download if set to False")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    dates = [date.strftime('%Y%m%d') for date in pd.date_range(start=start_date, end=end_date)]

    if args.download:
        # Step 1: Download all data
        for date in dates:
            logging.info(f"Downloading RRQPE data for {date}")
            download_rrqpe(args.rrqpepath, date)
    else:
        logging.info("Skipping download step")

    # Step 2: Gather all files
    logging.info("Gathering all files for processing")
    file_list = gather_files(args.rrqpepath, dates)

    # Step 3: Process all files
    logging.info("Processing all files")
    calculate_acum_rrqpe(file_list, args.extent, args.max_workers)
    
    # Step 4: Save results
    if aggregated_list:
        final_df = pd.concat(aggregated_list, ignore_index=True)
        final_df.to_parquet(args.destpath + '/rrqpe_data.parquet', index=False)
        final_df.to_csv(args.destpath + '/rrqpe_data.csv', index=False)
    else:
        logging.info("No data processed, aggregated list is empty.")

if __name__ == "__main__":
    main()
