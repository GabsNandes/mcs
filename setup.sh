#!/bin/bash

#Create env
conda env create -f env.yml

#Create data folder
mkdir -p ./data

# Activate env
source /opt/anaconda/bin/activate arboseer

# Download and process CNES Health Units data
python src/utils/download_cnes_file.py ST RJ 2401 data/raw/cnes/STRJ2311.dbc
python src/utils/dbc_to_parquet.py data/raw/cnes/STRJ2401.dbc data/raw/cnes/STRJ2401.parquet
python src/process_cnes_dataset.py data/raw/cnes/STRJ2401.parquet data/processed/cnes/STRJ2401.parquet

# Download and process SINAN dengue cases
python src/utils/download_sinan_file.py DENG 2023 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR23.parquet data/processed/sinan/DENGBR23.parquet --cod_uf 33

# Download inmet data
python src/utils/download_inmet_data.py -s A621 -b 2023 -e 2023 -o data/processed/inmet --api_token <>

# Calculate Lat/Lon lookup arrays
python src/utils/calculate_lats_lons.py ABI-L2-LSTF data/processed
python src/utils/calculate_lats_lons.py ABI-L2-RRQPEF data/processed

# Calculate LST
python src/calculate_min_max_avg_lst.py 20220101 20221231 data/raw/lst data/processed/lst
python src/calculate_min_max_avg_lst.py 20230101 20231231 data/raw/lst data/processed/lst

# Calculate RRQPE
 python src/calculate_accumulated_rrqpe.py 20230101 20231231 data/raw/rrqpe data/processed/rrqpe