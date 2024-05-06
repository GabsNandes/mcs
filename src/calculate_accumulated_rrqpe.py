import pandas as pd
from utils.download_rrqpe_data import download_rrqpe_data
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
acum = np.zeros((5424,5424))
# GLOBALS 

var = "RRQPE"

def download_rrqpe(path, date):
    download_rrqpe_data(date, f"{path}/{date}")

def process_file(file_path):
    global acum

    img = gdal.Open(f"NETCDF:{file_path}:{var}")
    dqf = gdal.Open(f"NETCDF:{file_path}:DQF")

    metadata = img.GetMetadata()
    scale = float(metadata.get(var + "#scale_factor"))
    offset = float(metadata.get(var + "#add_offset"))
    undef = float(metadata.get(var + "#_FillValue"))
    dtime = metadata.get("NC_GLOBAL#time_coverage_start")

    ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)
    ds_dqf = dqf.ReadAsArray(0, 0, dqf.RasterXSize, dqf.RasterYSize).astype(float)

    ds = (ds * scale + offset)
    ds[ds_dqf > 1] = np.nan

    acum += np.where(np.isnan(ds), 0, ds)

def calculate_accumulated_rrqpe(date, rrqpepath, destpath): 
    os.makedirs(destpath, exist_ok=True)  # Ensure output directory exists
    download_rrqpe(rrqpepath, date)

    # Init variables for a day
    acum.fill(0)

    for file in os.listdir(f"{rrqpepath}/{date}"):      
        file_path = os.path.join(f"{rrqpepath}/{date}", file)
        if os.path.isfile(file_path):
            process_file(file_path)     

    np.savez(f"{destpath}/{date}.npz", acum=acum)
    logging.info(f"Saved RRQPE data at {destpath}/{date}.npz")

def main():   
    parser = argparse.ArgumentParser(description="Calculate RRQPE data for a date range, saving NP arrays")
    parser.add_argument("start", help="Date start in yyyymmdd format")
    parser.add_argument("end", help="Date end in yyyymmdd format")
    parser.add_argument("rrqpepath", help="Path where rrqpe files will be downloaded, default is data/raw/rrqpe", default="data/raw/rrqpe")    
    parser.add_argument("destpath", help="Destination path, default is data/processed/rrqpe", default="data/processed/rrqpe")        
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    start_date = datetime.strptime(args.start, '%Y%m%d')
    end_date = datetime.strptime(args.end, '%Y%m%d')

    current_date = start_date

    while current_date <= end_date:
        logging.info(f"Calculating RRQPE for {current_date.strftime('%Y%m%d')}")

        calculate_accumulated_rrqpe(current_date.strftime('%Y%m%d'), args.rrqpepath, args.destpath)    
        current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
