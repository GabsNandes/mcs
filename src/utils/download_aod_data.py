from utils.download_goes_prod import download_goes_prod
import numpy as np
import logging
import argparse
from datetime import datetime
from datetime import timedelta
import os

def download_aod_data(date, path):
    product = "ABI-L2-AODF"
    minutes = [0, 10, 20, 30, 40, 50]  # Minutos específicos para download
    os.makedirs(path, exist_ok=True)  # Ensure output directory exists

    for hour in np.arange(0, 24-3, 1):
        for minute in minutes:
            yyyymmddhhmn = f"{date}{hour:02.0f}{minute:02.0f}"
            file_name = download_goes_prod(yyyymmddhhmn, product, path)

    for hour in np.arange(24-3, 24, 1):
        previous_day = datetime.strptime(date, '%Y%m%d')
        previous_day -= timedelta(days=1)
        for minute in minutes:
            yyyymmddhhmn = f"{previous_day.strftime('%Y%m%d')}{hour:02.0f}{minute:02.0f}"
            file_name = download_goes_prod(yyyymmddhhmn, product, path)

def main():
    parser = argparse.ArgumentParser(description="Download ABI-L2-AOD product from GOES-16 for a date")
    parser.add_argument("date", help="Date in yyyymmdd format")
    parser.add_argument("path", help="Destination path, default is data/raw", default="data/raw/aod")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="DEBUG", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    download_aod_data(args.date, args.path)

if __name__ == "__main__":
    main()
