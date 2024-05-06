import pandas as pd
from utils.download_adp_data import download_adp_data
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
dust = np.zeros((5424,5424))
smoke = np.zeros((5424,5424))
# GLOBALS 

dust_var = "Dust"
smoke_var = "Smoke"

def download_adp(path, date):
    download_adp_data(date, f"{path}/{date}")

def process_file(file_path):
    global dust, smoke

    dust_img = gdal.Open(f"NETCDF:{file_path}:{dust_var}")
    smoke_img = gdal.Open(f"NETCDF:{file_path}:{smoke_var}")    
    dqf = gdal.Open(f"NETCDF:{file_path}:DQF")

    metadata = dust_img.GetMetadata()
    undef = float(metadata.get(dust_var + "#_FillValue"))
    dtime = metadata.get("NC_GLOBAL#time_coverage_start")

    dust_ds = dust_img.ReadAsArray(0, 0, dust_img.RasterXSize, dust_img.RasterYSize).astype(float)
    dust_ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

    dust_ds[dust_ds_dqf > 1] = np.nan

    smoke_ds = smoke_img.ReadAsArray(0, 0, smoke_img.RasterXSize, smoke_img.RasterYSize).astype(float)
    smoke_ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

    smoke_ds[smoke_ds_dqf > 1] = np.nan

    dust_present = dust_ds > 0
    smoke_present = smoke_ds > 0 
    dust = np.logical_or(dust.astype(bool), dust_present).astype(int)
    smoke = np.logical_or(smoke.astype(bool), smoke_present).astype(int)    

def calculate_accumulated_adp(date, adppath, destpath): 
    os.makedirs(destpath, exist_ok=True)  # Ensure output directory exists
    download_adp(adppath, date)

    # Init variables for a day
    dust.fill(0)
    smoke.fill(0)

    for file in os.listdir(f"{adppath}/{date}"):      
        file_path = os.path.join(f"{adppath}/{date}", file)
        if os.path.isfile(file_path):
            process_file(file_path)     

    np.savez(f"{destpath}/{date}.npz", dust=dust, smoke=smoke)
    logging.info(f"Saved ADP data at {destpath}/{date}.npz")

def main():   
    parser = argparse.ArgumentParser(description="Calculate ADP data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("adppath", help="Path where adp files will be downloaded, default is data/raw/adp", default="data/raw/adp")    
    parser.add_argument("destpath", help="Destination path, default is data/processed/adp", default="data/processed/adp")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating adp for {current_date.strftime('%Y%m%d')}")

        calculate_accumulated_adp(current_date.strftime('%Y%m%d'), args.adppath, args.destpath)    
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
