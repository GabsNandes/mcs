#!/bin/bash
# Activate env
source /opt/anaconda/bin/activate arboseer

# Download and process SINAN dengue cases
python src/utils/download_sinan_file.py DENG 2019 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR19.parquet data/processed/sinan/DENGBR19.parquet --cod_uf 33
python src/utils/download_sinan_file.py DENG 2020 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR20.parquet data/processed/sinan/DENGBR20.parquet --cod_uf 33
python src/utils/download_sinan_file.py DENG 2021 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR21.parquet data/processed/sinan/DENGBR21.parquet --cod_uf 33
python src/utils/download_sinan_file.py DENG 2022 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR22.parquet data/processed/sinan/DENGBR22.parquet --cod_uf 33
python src/utils/download_sinan_file.py DENG 2023 data/raw/sinan
python src/extract_sinan_cases.py data/raw/sinan/DENGBR23.parquet data/processed/sinan/DENGBR23.parquet --cod_uf 33

# Download inmet data
python src/utils/download_inmet_data.py -s A621 -b 2019 -e 2023 -o data/processed/inmet --api_token ejlJQjBhRWw0bUlUNlBnY0taRWFjWnFoSExSVFUwNW4=z9IB0aEl4mIT6PgcKZEacZqhHLRTU05n
python src/utils/download_inmet_data.py -s A636 -b 2019 -e 2023 -o data/processed/inmet --api_token ejlJQjBhRWw0bUlUNlBnY0taRWFjWnFoSExSVFUwNW4=z9IB0aEl4mIT6PgcKZEacZqhHLRTU05n