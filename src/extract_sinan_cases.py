import logging
import argparse
import pandas as pd
import os

def extract_sinan_cases(cnes_id, cod_uf, input_path, output_path):
    """
    Extract cases from a SINAN parquet file

    Args:
    id: str, CNES number
    input_path: str, Parquet file path
    output_path: str, Destination path

    """        
    try:
        df = pd.read_parquet(input_path)
        if cnes_id is not None:
            df = df[df["ID_UNIDADE"] == cnes_id]
        if cod_uf is not None:
            df = df[df["SG_UF"] == cod_uf]

        grouped_df = df.groupby(["ID_AGRAVO", "ID_UNIDADE", "DT_NOTIFIC"])
        aggregated_df = grouped_df.size().reset_index(name="count")

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

        aggregated_df.to_parquet(output_path, index=False)

        logging.info(f"Cases extracted at: {output_path}")    
    except Exception as e:
        logging.error(f"Erro extracting cases: {e}")    
 
def main():
    parser = argparse.ArgumentParser(description="Extract cases from a SINAN parquet file")
    parser.add_argument("input_path", help="Path to the input Parquet file")
    parser.add_argument("output_path", help="Path to the output Parquet file")    
    parser.add_argument("--cnes_id", dest="cnes_id", help="CNES number", default=None)    
    parser.add_argument("--cod_uf", dest="cod_uf", help="UF code of the cases", default=None)    
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="INFO", help="Set the logging level")

    args = parser.parse_args()    

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")   
    
    if args.output_path is None:
        args.output_path = args.input_path
    
    extract_sinan_cases(args.cnes_id, args.cod_uf, args.input_path, args.output_path)

if __name__ == "__main__":
    main()