import pandas as pd
from utils.download_lst_data import download_lst_data
import logging
import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import argparse
from osgeo import gdal

# GLOBALS 
sum_ds = np.zeros((1086,1086))
count_ds = np.zeros((1086,1086))
min_ds = np.full((1086,1086), np.inf)
max_ds = np.full((1086,1086), -np.inf)
# GLOBALS 

var = "LST"

def download_lst(path, date):
    download_lst_data(date, f"{path}/{date}")

def process_file(file_path):
    global sum_ds, count_ds, min_ds, max_ds

    img = gdal.Open(f"NETCDF:{file_path}:{var}")
    dqf = gdal.Open(f"NETCDF:{file_path}:DQF")

    metadata = img.GetMetadata()
    scale = float(metadata.get(var + "#scale_factor"))
    offset = float(metadata.get(var + "#add_offset"))
    undef = float(metadata.get(var + "#_FillValue"))
    dtime = metadata.get("NC_GLOBAL#time_coverage_start")

    ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)
    ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

    ds = (ds * scale + offset) - 273.15
    ds[ds_dqf > 1] = np.nan

    min_ds = np.fmin(min_ds, np.nan_to_num(ds, nan=np.inf))
    max_ds = np.fmax(max_ds, np.nan_to_num(ds, nan=-np.inf))
    sum_ds += np.where(np.isnan(ds), 0, ds)
    count_ds += np.where(np.isnan(ds), 0, 1)

def calculate_min_max_avg_lst(date, lstpath, destpath): 
    os.makedirs(destpath, exist_ok=True)  # Ensure output directory exists
    download_lst(lstpath, date)

    # Init variables for a day
    sum_ds.fill(0)
    count_ds.fill(0)
    min_ds.fill(np.inf)
    max_ds.fill(-np.inf)

    for file in os.listdir(f"{lstpath}/{date}"):      
        file_path = os.path.join(f"{lstpath}/{date}", file)
        if os.path.isfile(file_path):
            process_file(file_path)     

    avg_ds = np.divide(sum_ds, count_ds, out=np.full_like(sum_ds, np.nan), where=count_ds!=0)

    day_ds = [avg_ds, min_ds, max_ds]

    np.savez(f"{destpath}/{date}.npz", avg=avg_ds, min=min_ds, max=max_ds)
    logging.info(f"Saved LST data at {destpath}/{date}.npz")

def main():   
    parser = argparse.ArgumentParser(description="Calculate LST data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("lstpath", help="Path where lst files will be downloaded, default is data/raw", default="data/raw")    
    parser.add_argument("destpath", help="Destination path, default is data/raw", default="data/raw")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating LST for {current_date.strftime('%Y%m%d')}")

        calculate_min_max_avg_lst(current_date.strftime('%Y%m%d'), args.lstpath, args.destpath)    
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
