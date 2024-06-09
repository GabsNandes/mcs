import pandas as pd 

pd.read_parquet('data/raw/sinan/DENGBR20.parquet').head(100).to_csv('teste.csv')