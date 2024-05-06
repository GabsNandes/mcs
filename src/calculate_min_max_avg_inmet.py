import pandas as pd
import os
import logging
import argparse

def calculate_min_max_avg_inmet(original_path, dest_name):   
    original_data = pd.read_parquet(f"{original_path}/concat.parquet")

    original_data.to_csv('teste.csv')

def main():   
    calculate_min_max_avg_inmet('data/processed/inmet', 'teste')    

if __name__ == "__main__":
    main()
