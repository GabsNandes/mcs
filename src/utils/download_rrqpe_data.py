from utils.download_goes_prod import download_goes_prod
import numpy as np
import logging
import argparse
from datetime import datetime
from datetime import timedelta
import os

def download_rrqpe_data(date, path):
    product = "ABI-L2-RRQPEF"
    os.makedirs(path, exist_ok=True)
    for hour in np.arange(0,21,1):
        for minute in range(0, 60, 10):
            yyyymmddhhmn = f"{date}{hour:02.0f}{minute:02.0f}"
            download_goes_prod(yyyymmddhhmn, product, path)  

    previous_day = datetime.strptime(date, '%Y%m%d') - timedelta(days=1)
    for hour in np.arange(21,24,1):       
        for minute in range(0, 60, 10):
            yyyymmddhhmn = f"{previous_day.strftime('%Y%m%d')}{hour:02.0f}{minute:02.0f}"
            download_goes_prod(yyyymmddhhmn, product, path)

def main():
    parser = argparse.ArgumentParser(description="Download ABI-L2-RRQPE product from GOES-16 for a date")
    parser.add_argument("date", help="Date in yyyymmdd format")
    parser.add_argument("path", help="Destination path, default is data/raw", default="data/raw/rrqpe")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="DEBUG", help="Set the logging level")
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    download_rrqpe_data(args.date, args.path)

if __name__ == "__main__":
    main()