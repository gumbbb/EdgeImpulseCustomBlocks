import pandas as pd
import os
import glob

def get_base_path():
    """
    Auto detect base path:
    - For Edge Impulse → use bucket path
    - For local → use path local
    """
    bucket_name = os.getenv('BUCKET_NAME')
    bucket_directory = os.getenv('BUCKET_DIRECTORY')
    mount_prefix = os.getenv('MOUNT_PREFIX', '/mnt/s3fs/')
    if bucket_name:
        base_path = os.path.join(mount_prefix, bucket_name)
        if bucket_directory:
            base_path = os.path.join(base_path, bucket_directory)
        print(f"Using bucket path: {base_path}", flush=True)
    else:
        base_path = "/home/anhtnt18/TMC_Qualcomm/data/"
        print(f"Using local path: {base_path}", flush=True)
    return base_path
 

BASE_PATH = get_base_path()

config_data = {
    "base_directories": {
        "raw_data": os.path.join(BASE_PATH, "raw_data/"),
        "clean_data": os.path.join(BASE_PATH, "clean_data/"),
        "monitoring_data": os.path.join(BASE_PATH, "monitoring_data/"),
    },
    "raw_data_directories": [
        "20250601-0614_20units/"
    ],
    "pcs_raw_data_directories": ["20250601-0615pcs_data/"],
}
# raw data
pcs_raw_data_directories = config_data.get("pcs_raw_data_directories")
raw_data_path = config_data.get("base_directories", []).get("raw_data")
raw_data_directories = config_data.get("raw_data_directories")

def get_current_pcs_raw_data() -> pd.DataFrame:

    df = pd.DataFrame()

    if pcs_raw_data_directories:
        for raw_dir in pcs_raw_data_directories:

            print(raw_dir)
            file_list = glob.glob(raw_data_path + raw_dir + "*.csv.gz")

            [print("- " + os.path.basename(f)) for f in file_list]

            # dfs = [pd.read_csv(f) for f in file_list]
            # df = pd.concat(dfs, ignore_index=True)
            df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True, copy=False) 
    return df

def get_current_raw_data() -> pd.DataFrame:
    if raw_data_directories:
        for raw_dir in raw_data_directories:
            print(raw_dir)
            file_list = glob.glob(raw_data_path + raw_dir + "*.csv.gz")
            [print("- " + os.path.basename(f)) for f in file_list]
            print("Starting read VDP data")
            # Check actual uncompressed size
            first_df = pd.read_csv(file_list[0])
            print(f"Memory usage of first file: {first_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Shape: {first_df.shape}")
            del first_df
            # Optimize dtypes while reading
            df = pd.concat([pd.read_csv(f, low_memory=False) for f in file_list], 
                          ignore_index=True, copy=False)
            print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return df
    return pd.DataFrame()