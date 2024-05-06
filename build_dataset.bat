@echo off

:: Activate env
call /opt/anaconda/bin/activate.bat arboseer

:: Download and process CNES Health Units data
python src\utils\download_cnes_file.py ST RJ 2401 data\raw\cnes\STRJ2311.dbc
python src\utils\dbc_to_parquet.py data\raw\cnes\STRJ2401.dbc data\raw\cnes\STRJ2401.parquet
python src\process_cnes_dataset.py data\raw\cnes\STRJ2401.parquet data\processed\cnes\STRJ2401.parquet

:: Download and process SINAN dengue cases
python src\utils\download_sinan_file.py DENG 2023 data\raw\sinan
python src\extract_sinan_cases.py data\raw\sinan\DENGBR23.parquet data\processed\sinan\DENGBR23.parquet --cod_uf 33

:: Download inmet data
python src\utils\download_inmet_data.py -s A617 -b 2023 -e 2023 -o data\processed\inmet --api_token %INMET_API_TOKEN%
python src\utils\download_inmet_data.py -s A615 -b 2023 -e 2023 -o data\processed\inmet --api_token %INMET_API_TOKEN%
python src\utils\download_inmet_data.py -s A628 -b 2023 -e 2023 -o data\processed\inmet --api_token %INMET_API_TOKEN%
:: Repeat for all remaining stations...

:: Concat inmet data
python src\unify_inmet.py data\raw\lst data\raw\inmet data\processed\inmet

:: Calculate inmet data
python src\calculate_min_max_avg_inmet.py data\processed\inmet

:: Build
python src\build_dataset.py
