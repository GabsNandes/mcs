#!/bin/bash

#Create env
conda env create -f arboseer.yml

#Create data folder
mkdir -p ./data

# Activate env
conda activate arboseer

# Download and process CNES Health Units data
./build_dataset.sh