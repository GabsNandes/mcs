import pandas as pd
from utils.download_aod_data import download_aod_data
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
aod_accum = np.zeros((5424,5424))
# GLOBALS 

var = "AOD"

def download_aod(path, date):
    download_aod_data(date, f"{path}/{date}")

def process_file(file_path):
    global aod_accum

    img = gdal.Open(f"NETCDF:{file_path}:{var}")
    dqf = gdal.Open(f"NETCDF:{file_path}:DQF")

    metadata = img.GetMetadata()
    undef = float(metadata.get(var + "#_FillValue"))
    dtime = metadata.get("NC_GLOBAL#time_coverage_start")

    ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)
    ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

    ds[ds_dqf > 1] = np.nan

    ds_present = ds > 0
    aod_accum = np.logical_or(aod_accum.astype(bool), ds_present).astype(int)

def calculate_accumulated_aod(date, aodpath, destpath): 
    os.makedirs(destpath, exist_ok=True)  # Ensure output directory exists
    download_aod(aodpath, date)

    # Init variables for a day
    aod_accum.fill(0)

    for file in os.listdir(f"{aodpath}/{date}"):      
        file_path = os.path.join(f"{aodpath}/{date}", file)
        if os.path.isfile(file_path):
            process_file(file_path)     

    np.savez(f"{destpath}/{date}.npz", aod=aod_accum)
    logging.info(f"Saved AOD data at {destpath}/{date}.npz")

def main():   
    parser = argparse.ArgumentParser(description="Calculate AOD data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("aodpath", help="Path where aod files will be downloaded, default is data/raw/aod", default="data/raw/aod")    
    parser.add_argument("destpath", help="Destination path, default is data/processed/aod", default="data/processed/aod")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating aod for {current_date.strftime('%Y%m%d')}")

        calculate_accumulated_aod(current_date.strftime('%Y%m%d'), args.aodpath, args.destpath)    
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
