# Globals
#-----------------------------------------------------------------------------------------------------------
from datetime import datetime
from datetime import timedelta
import os
import logging
import numpy as np
from src.utils.download_lst_data import download_lst_data
from src.utils.reproject_nc_file import reproject
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler') 
import re
from netCDF4 import Dataset
import random

logging.basicConfig(level=getattr(logging, "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")    

start_date = '20230101'
end_date = '20231231'
processed_lst_path_gaps = 'data'
raw_path = processed_lst_path_gaps+'/raw'
raw_lst_path = raw_path+'/lst'
processed_path = processed_lst_path_gaps+'/processed'
processed_lst_path = processed_path+'/lst'
processed_lst_path_gaps = processed_path+'/lst_gaps'
processed_misc_path = processed_lst_path+'/misc'


loni, lonf, lati, latf = -50.60, -22.60, -44.00, -16.00
extent = [loni, lonf, lati, latf]

#-----------------------------------------------------------------------------------------------------------

# First, download the LST data
#-----------------------------------------------------------------------------------------------------------
def download_goes():
    start_date_formated = datetime.strptime(start_date, '%Y%m%d')
    end_date_formated = datetime.strptime(end_date, '%Y%m%d')
    current_date = start_date_formated
    os.makedirs(raw_lst_path, exist_ok=True)

    while current_date <= end_date_formated:
        logging.info(f"Downloading LST data for {current_date.strftime('%Y%m%d')}")
        download_lst_data(current_date.strftime('%Y%m%d'), f"{raw_lst_path}")
        current_date += timedelta(days=1)

    logging.info(f"Done downloading LST data")    
#-----------------------------------------------------------------------------------------------------------

# Now, reproject the data and create the array mask for places that never have value
#-----------------------------------------------------------------------------------------------------------
def reproject_files():
    os.makedirs(processed_lst_path, exist_ok=True)
    os.makedirs(processed_misc_path, exist_ok=True)
    mask_array = None

    files = os.listdir(raw_lst_path)
    files.sort(key=lambda x: datetime.strptime(re.search(r'_c(\d{13})', x).group(1), "%Y%j%H%M%S"))    

    for raw_lst_file in files:
        raw_lst_file_path = os.path.join(raw_lst_path, raw_lst_file)
        if os.path.isfile(raw_lst_file_path):
            # Load the metadata
            logging.info(f"Reprojecting file: {raw_lst_file}")
            lst_variables = gdal.Open(f'NETCDF:{raw_lst_file_path}:LST')
            dqf_variables = gdal.Open(f'NETCDF:{raw_lst_file_path}:DQF')
            metadata = lst_variables.GetMetadata()
            scale = float(metadata.get('LST#scale_factor'))
            offset = float(metadata.get('LST#add_offset'))
            undef = float(metadata.get('LST#_FillValue'))
            dtime = metadata.get('NC_GLOBAL#time_coverage_start')
            data_formatada = datetime.strptime(dtime, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y%m%d%H%M") 

            # Load the data
            lst_data = lst_variables.ReadAsArray(0, 0, lst_variables.RasterXSize, lst_variables.RasterYSize).astype(float)
            dqf_data = dqf_variables.ReadAsArray(0, 0, dqf_variables.RasterXSize, dqf_variables.RasterYSize).astype(float)

            # Apply the scale, offset and convert to celsius
            lst_data = (lst_data * scale + offset) - 273.15

            # Apply NaN's where the quality flag is greater than 1
            lst_data[dqf_data > 1] = np.nan

            if mask_array is None:
                mask_array = np.zeros_like(lst_data, dtype=int)
           
            reprojected_filename = f'{processed_lst_path}/{data_formatada}.nc'
            reproject(reprojected_filename, lst_variables, lst_data, extent, undef)

            mask_array[~np.isnan(lst_data)] = 1
    
    logging.info(f"Done reprojecting LST files")

    reprojected_filename = f'{processed_misc_path}/mask.nc'
    reproject(reprojected_filename, lst_variables, mask_array, extent, undef)

    logging.info(f"Saved LST info {processed_misc_path}/mask.nc")
#-----------------------------------------------------------------------------------------------------------        

# After reprojecting the files, we will introduce gaps
#-----------------------------------------------------------------------------------------------------------
def generate_gaps():
    pass

#-----------------------------------------------------------------------------------------------------------

# Finally, execute everything
#-----------------------------------------------------------------------------------------------------------
def main():
    download_goes()
    reproject_files()
    generate_gaps()

if __name__ == "__main__":
    main()