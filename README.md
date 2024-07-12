
# mcs
Climate Change on Health

## About
Soon

## Features
Soon

## Installation
```bash
conda env create -f arboseer.yml
conda activate arboseer
```

## Building the dataset
The dataset used to train the LSTM model is composed of:
- CNES - Dataset of Health Units
- SINAN - Dataset of Dengue Cases
- INMET - Dataset of INMET weather measurements
- LST - Dataset of GOES-16 Land Surface Temperature
- RRQPE - Dataset of GOES-16 Rainfal Rate

The full script for building the dataset can be found in the file build_dataset.sh/bat.

Note that raw files aren't removed after processing.

### Download and process CNES Health Units data 
In this step, we'll download the latest CNES data, convert it to parquet and add the lat/lon values for the addresses.

The first script downloads the data.
```bash
python src/utils/download_cnes_file.py FILETYPE UF DATE DEST_PATH/FILENAME.dbc
```

The next one converts it from dbc to parquet.
```bash
python src/utils/dbc_to_parquet.py INPUT_PATH OUTPUT_PATH
```

The final script adds the lat/lon values and trims the dataset fields.
```bash
python src/process_cnes_dataset.py INPUT_PATH OUTPUT_PATH
```

Upon finishing this step we should have the final CNES parquet.

### Download and process SINAN dengue cases
In this step, we'll download the cases using PySus and extract the SINAN cases. PySus only works on Linux. The files can also be acquired on the SINAN website.

First, we'll download the file for every disease and year.
```bash
python src/utils/download_sinan_file.py FILETYPE YEAR OUTPUT_PATH
```

Then, we'll merge the files into a single dataset. As of now, the dataset has a fixed name: concat.parquet
```bash
python src/unify_sinan.py INPUT_PATH OUTPUT_PATH
```

Finally, we'll extract the cases we want, trimming the dataset fields in the process. Here we have the option to filter by UF (with --cod_uf: RJ is 33, SP is 35 and so on) or CNES (with --cnes_id). We also can fill the dataset, inserting rows with 0 cases on the dates that aren't present on the dataset.
```bash
python src/extract_sinan_cases.py INPUT_PATH OUTPUT PATH --cod_uf COD_UF --filled --start_date YYYY-MM-DD --end_date YYYY-MM-DD
```

### Download INMET data    
In this step, we'll download and unify the INMET data. You'll need an INMET API token to make requests.
```bash
python src/utils/download_inmet_data.py -s STATION -b YYYY -e YYYY -o OUTPUT_PATH --api_token INMET_API_TOKEN
```

### Concat INMET data
With the INMET data downloaded, we'll unify all the files into a single dataset. We can also use --aggregated True to aggregate the values from an hourly basis to a daily basis.
```bash
python src/unify_inmet.py INPUT_PATH OUTPUT_PATH --aggregated True
```

### Calculate LST
In this step, we'll be downloading LST data and converting it into a single dataset. Right now we're using a fixed extent around Rio de Janeiro and a fixed output file name lst.parquet. In this process, we also aggregate measurements from hourly to daily creating the MIN, MAX, and AVG of each temperature.
```bash
python src/calculate_min_max_avg_lst.py YYYYMMDD YYYYMMDD DOWNLOAD_PATH OUTPUT_PATH
```

### Download and Calculate RRQPE
In this step, we'll be downloading RRQPE data and converting it into a single dataset. Right now we're using a fixed extent around Rio de Janeiro and a fixed output file name rrqpe.parquet. In this process, we also aggregate measurements from hourly to daily creating the SUM of the hourly rainfall rate.
```bash
python src/calculate_accumulated_rrqpe.py YYYYMMDD YYYYMMDD DOWNLOAD_PATH OUTPUT_PATH
```

### Build
Finally, we'll combine all the data into a single dataset.
```bash
python src/build_dataset.py DENG_DATASET CNES_DATASET INMET_DATASET LST_DATASET RRQPE_DATASET OUTPUTDATASET --start_date YYYY-MM-DD --end_date YYYY-MM-DD
```

## Train the LSTM model
To train the LSTM model, we simply call the train script passing the path to the built dataset, the output path (for figures and etc), and the date which will be used to split the data into test/train.
```bash
python src/train_lstm.py INPUT_PATH OUTPUT_PATH YYYY-MM-DD
```
