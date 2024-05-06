import boto3                             
from botocore import UNSIGNED            
from botocore.config import Config       
import logging
import os
from datetime import datetime

def download_goes_prod(yyyymmddhhmn, product_name, path_dest):
  """
  Download product from AMAZON AWS GOES-16 repo

  Args:
  yyyymmddhhmn: str, date
  product_name: str, product name
  path_dest: str, Destination path

  """      

  os.makedirs(path_dest, exist_ok=True)

  year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%Y')
  day_of_year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%j')
  hour = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%H')
  min = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%M')

  # AMAZON repository information
  # https://noaa-goes16.s3.amazonaws.com/index.html
  bucket_name = 'noaa-goes16'

  # Initializes the S3 client
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  #-----------------------------------------------------------------------------------------------------------
  # File structure
  prefix = f'{product_name}/{year}/{day_of_year}/{hour}/OR_{product_name}-M6_G16_s{year}{day_of_year}{hour}{min}'

  # Seach for the file on the server
  s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter = "/")

  #-----------------------------------------------------------------------------------------------------------
  # Check if there are files available
  if 'Contents' not in s3_result:
    # There are no files
    logging.error(f'No files found for the date: {yyyymmddhhmn}, Product-{product_name}')
    return -1
  else:
    # There are files
    for obj in s3_result['Contents']:
      key = obj['Key']
      # Print the file name
      file_name = key.split('/')[-1].split('.')[0]

      # Download the file
      if os.path.exists(f'{path_dest}/{file_name}.nc'):
        logging.info(f'File {path_dest}/{file_name}.nc exists')
      else:
        logging.info(f'Downloading file {path_dest}/{file_name}.nc')
        s3_client.download_file(bucket_name, key, f'{path_dest}/{file_name}.nc')
  return f'{file_name}'

def main():
    parser = argparse.ArgumentParser(description="Download GOES-16 product")
    parser.add_argument("date", help="File date in yyyymmddhhmn format")
    parser.add_argument("product_name", help="GOES-16 product, default is ABI-L2-LSTF", default="ABI-L2-LSTF")
    parser.add_argument("path_dest", help="Destination path, default is data/raw/goes", default="data/raw/goes")
    parser.add_argument("--log", dest="log_level", choices=["INFO", "DEBUG", "ERROR"], default="DEBUG", help="Set the logging level")
    
    args = parser.parse_args()
        
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")    

    download_goes_prod(args.date, args.product_name, args.path_dest)

if __name__ == "__main__":
    main()
      