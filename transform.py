"""
Edge Impulse Transformation Block for CAN Data Anomaly Detection.
 
Pipeline:
    Raw .csv.gz -> Preprocessing (1Hz)
               -> Feature Engineering (add_all_features)
               -> Windowing + Labeling (AnomalyDataGenerator)
               -> Flatten (prepare_data: GROUP BY mean, TNS one-hot)
               -> CSV Output
 
Input:  Raw .csv.gz files from configured directories (PCS + VDP)
Output: Flattened samples CSV (1 row per sample, 14 features + label)
"""
import errno
import os
import pandas as pd
import numpy as np
import argparse
import logging
import glob
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple, Any
from tqdm import tqdm
import tempfile
 
from src.preprocessing import preprocess_data
from src.gen_data import AnomalyDataGenerator
from src.utils.model_utils import prepare_data as ml_prepare_data
 
MAX_CSV_SIZE = 50 * 1024 * 1024
MAX_CSV_ROWS = 200000
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
DATA_CONFIG = {
    "raw_data_subdir": "raw_data/",
    "vdp_directories": [
        "20250601-0614_20units/"
    ],
    "pcs_directories": [
        "20250601-0615pcs_data/"
    ]
}
 
def safe_copy(src, dst, retries=5, sleep=2):
    """Safely copy a file from src to dst with retries on I/O errors."""
    for i in range(retries):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            return
        except OSError as e:
            if e.errno in (errno.EIO, errno.ETIMEDOUT):
                logger.warning("Copy failed (%s); retry %d/%d", e, i + 1, retries)
                time.sleep(sleep * (i + 1))
                continue
            raise
 
 
def save_metadata(metadata, out_directory):
    """Save Edge Impulse metadata JSON and exit."""
    all_ok = all(metadata['metadata'][m] != 0 for m in metadata['metadata'])
    metadata['metadata']['ei_check'] = 1 if all_ok else 0
 
    if out_directory is None:
        out_directory = os.getenv('EI_OUTPUT_DIR', '/tmp/out')
        os.makedirs(out_directory, exist_ok=True)
 
    metadata_path = os.path.join(out_directory, 'ei-metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info("Metadata saved: %s", metadata_path)
    exit(0)
 
 
def split_dataframe_by_size(df: pd.DataFrame, max_size_bytes: int = MAX_CSV_SIZE,
                            max_rows: int = MAX_CSV_ROWS) -> List[pd.DataFrame]:
    """Split DataFrame into chunks that respect size and row limits."""
    if len(df) == 0:
        return [df]
 
    sample_csv = df.head(100).to_csv(index=False)
    bytes_per_row = len(sample_csv.encode('utf-8')) / min(100, len(df))
    if bytes_per_row == 0:
        bytes_per_row = 1
 
    row_limit_by_size = int(max_size_bytes / bytes_per_row * 0.9)
    chunk_size = min(max_rows, row_limit_by_size)
   
    print(f"DEBUG: Splitting DataFrame - estimated {bytes_per_row:.0f} bytes/row, chunk_size={chunk_size}", flush=True)
 
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    return chunks
 
 
def detect_input_path(args) -> Optional[str]:
    """Detect input data path using tiered fallback strategy."""
    input_base_path = None
 
    # Tier 1: Argument-based
    if args.bucket_name:
        # Local path check (for Windows/Local testing)
        if os.path.exists(args.bucket_name):
            input_base_path = args.bucket_name if args.bucket_name.endswith(os.sep) or args.bucket_name.endswith('/') else args.bucket_name + os.sep
        else:
            azure_path = f"/mnt/azure/{args.bucket_name}"
            s3_path = f"/mnt/s3fs/{args.bucket_name}"
            if os.path.exists(azure_path):
                input_base_path = azure_path + "/"
            elif os.path.exists(s3_path):
                input_base_path = s3_path + "/"
 
    # Tier 2: Environment variables
    if not input_base_path:
        bucket_name = os.getenv('EI_BUCKET_NAME')
        data_path = os.getenv('EI_DATA_PATH')
        if bucket_name:
            azure_path = f"/mnt/azure/{bucket_name}"
            s3_path = f"/mnt/s3fs/{bucket_name}"
            if os.path.exists(azure_path):
                input_base_path = azure_path + "/"
            elif os.path.exists(s3_path):
                input_base_path = s3_path + "/"
 
        if not input_base_path and data_path and os.path.exists(data_path):
            input_base_path = data_path if data_path.endswith('/') else data_path + "/"
 
    # Tier 3: Common mount points scanning
    if not input_base_path:
        common_bases = ["/mnt/azure", "/mnt/s3fs"]
        print("DEBUG: Scanning common mount points...", flush=True)
        for base in common_bases:
            if not os.path.exists(base):
                continue
            try:
                subdirs = [d for d in os.listdir(base)
                           if os.path.isdir(os.path.join(base, d))]
                if subdirs:
                    input_base_path = os.path.join(base, subdirs[0]) + "/"
                    break
            except Exception as e:
                print(f"DEBUG: Error reading {base}: {e}", flush=True)
 
    return input_base_path
 
def get_files_from_specific_dirs(base_path: str, sub_dirs: List[str]) -> List[str]:
    """Locate .csv.gz files in specific subdirectories."""
    file_list = []
    raw_data_root = os.path.join(base_path, DATA_CONFIG["raw_data_subdir"])
 
    if not os.path.exists(raw_data_root):
        logger.warning("Raw data root not found at %s, falling back to base path", raw_data_root)
        raw_data_root = base_path
 
    for sub_dir in sub_dirs:
        search_path = os.path.join(raw_data_root, sub_dir, "*.csv.gz")
        found_files = glob.glob(search_path)
        logger.info("Directory %s -> %d file(s)", sub_dir, len(found_files))
        file_list.extend(found_files)
 
    return file_list
 
def preprocess_and_combine_files(csv_files: List[str], temp_dir: str,
                                 data_type: str) -> Optional[str]:
    """Preprocess a list of files and combine into one temp CSV.
 
    Calls src.preprocessing.preprocess_data which handles:
      - create_trip_id (Hased_VIN + TRIP_COUNT)
      - resampling to 1Hz
      - NaN row removal
    """
    if not csv_files:
        logger.info("No %s files to preprocess.", data_type)
        return None
 
    logger.info("Loading %d %s file(s)...", len(csv_files), data_type)
    all_dfs = []
    for f in tqdm(csv_files, desc=f"Loading {data_type}"):
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                all_dfs.append(df)
            except Exception as e:
                logger.error("Failed to read %s: %s", f, e)
 
    if not all_dfs:
        logger.error("Could not load any valid data for %s.", data_type)
        return None
 
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info("Combined raw %s data: %d rows, %d cols",
                data_type, len(combined_df), len(combined_df.columns))
 
    output_path = os.path.join(temp_dir, f"{data_type.lower()}_preprocessed.csv")
    try:
        preprocess_data(combined_df, output_path)
        logger.info("Preprocessing complete for %s -> %s", data_type, output_path)
    except Exception as e:
        logger.error("Preprocessing failed for %s: %s", data_type, e, exc_info=True)
        return None
 
    return output_path
 
def generate_samples(pcs_path: Optional[str], vdp_path: Optional[str],
                     temp_dir: str, window_size: int = 30,
                     num_negative_samples: int = 8) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Generate windowed samples using AnomalyDataGenerator.
 
    Internally calls add_all_features() to compute engineered features
    (times, avg_sudden_acceleration_count, TTC_filled, etc.) before windowing.
    """
    if not pcs_path and not vdp_path:
        logger.warning("No data source (PCS or VDP) found to generate samples.")
        return None
 
    try:
        generator = AnomalyDataGenerator(
            pcs_data_path=pcs_path,
            gen_data_path=vdp_path,
            output_dir=temp_dir,
            window_size=window_size,
            skip_window_size=10,
            random_state=42,
            num_negative_samples=num_negative_samples
        )
 
        generator.load_data()
        generator.build_trip_dictionaries()
        generator.downsample_gen_trips_to_match_pcs(seed=42)
       
        train_samples, test_samples = generator.generate_all_samples()
 
        if not train_samples and not test_samples:
            logger.warning("No samples generated from AnomalyDataGenerator.")
            return None, None
 
        train_df = pd.concat(train_samples, ignore_index=True) if train_samples else None
        test_df = pd.concat(test_samples, ignore_index=True) if test_samples else None
       
        logger.info("Generated %d train windows, %d test windows",
                     len(train_samples), len(test_samples))
 
        if train_df is not None and 'label' in train_df.columns and not train_df.empty:
            label_counts = train_df['label'].value_counts()
            logger.info("Train Window label distribution: %s", label_counts.to_dict())
           
        if test_df is not None and 'label' in test_df.columns and not test_df.empty:
            label_counts = test_df['label'].value_counts()
            logger.info("Test Window label distribution: %s", label_counts.to_dict())
 
        return train_df, test_df
 
    except Exception as e:
        logger.error("Sample generation failed: %s", e, exc_info=True)
        return None, None
 
def apply_prepare_data(samples_df: pd.DataFrame,
                       seq_len: int = 30,
                       skip_times: int = 10) -> Optional[pd.DataFrame]:
    """Convert windowed samples (Long Format) to flat feature vectors (Flat Format).
 
    Applies the same preprocessing as the LGBM training pipeline:
      1. Remove last (skip_times + 1) rows per trip to prevent data leakage
      2. Fill NaN with 0
      3. Call prepare_data: select tail(seq_len) rows per trip,
         select 14 features, GROUP BY new_trip_id with mean aggregation
 
    Input shape:  (num_samples * ~31, ~30 cols) - Long Format
    Output shape: (num_samples, 14 features + label) - Flat Format
    """
    df = samples_df.copy()
 
    skip_n = skip_times + 1
    df = df.sort_values(['new_trip_id', 'times'])
    group_sizes = df.groupby('new_trip_id', sort=False).transform('size')
    cumcounts = df.groupby('new_trip_id', sort=False).cumcount()
    df = df[cumcounts < (group_sizes - skip_n)].reset_index(drop=True)
 
    df = df.fillna(0)
 
    X, y, feature_names, meta = ml_prepare_data(df, seq_len=seq_len)
 
    flat_df = pd.DataFrame(X, columns=feature_names)
    flat_df['label'] = y.astype(int)
    flat_df['original_trip_id'] = meta
 
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    logger.info("Flattened: %d samples x %d features (pos=%d, neg=%d)",
                len(flat_df), len(feature_names), pos_count, neg_count)
 
    return flat_df
 
def process_datasets(pcs_files: List[str], vdp_files: List[str],
                     output_directory: str,
                     window_size: int = 30,
                     num_negative_samples: int = 8,
                     seq_len: int = 30,
                     skip_times: int = 3) -> Tuple[int, int]:
    """Full processing pipeline: Preprocess -> Window -> Flatten -> Save."""
    total_files = len(pcs_files) + len(vdp_files)
    logger.info("Processing %d file(s) (%d PCS, %d VDP)",
                total_files, len(pcs_files), len(vdp_files))
   
    logger.info("Pipeline parameters: window_size=%ds, num_negative_samples=%d, seq_len=%d, skip_times=%d",
                window_size, num_negative_samples, seq_len, skip_times)
   
    with tempfile.TemporaryDirectory(prefix="ei_transform_") as temp_dir:
        pcs_path = preprocess_and_combine_files(pcs_files, temp_dir, 'PCS')
        vdp_path = preprocess_and_combine_files(vdp_files, temp_dir, 'VDP')
 
        train_df, test_df = generate_samples(
            pcs_path, vdp_path, temp_dir, window_size, num_negative_samples
        )
 
        if (train_df is None or len(train_df) == 0) and (test_df is None or len(test_df) == 0):
            logger.error("No samples were generated. Pipeline halted.")
            return 0, total_files
           
        saved_count = 0
       
        # Process and save Train
        if train_df is not None and len(train_df) > 0:
            flat_train = apply_prepare_data(train_df, seq_len=seq_len, skip_times=skip_times)
            if flat_train is not None and len(flat_train) > 0:
                train_dir = os.path.join(output_directory, "training")
                os.makedirs(train_dir, exist_ok=True)
                chunks = split_dataframe_by_size(flat_train, MAX_CSV_SIZE, MAX_CSV_ROWS)
                logger.info("Train output split into %d chunk(s)", len(chunks))
               
                next_part_num = 1
                for chunk in chunks:
                    while True:
                        output_filename = f"prepared_samples_part{next_part_num:03d}.csv"
                        output_path = os.path.join(train_dir, output_filename)
                        if not os.path.exists(output_path):
                            break
                        next_part_num += 1
         
                    chunk.to_csv(output_path, index=False)
                    file_size = os.path.getsize(output_path)
                    logger.info("Saved %s in training/ (%d rows, %.2f MB)",
                                output_filename, len(chunk), file_size / 1024 / 1024)
                    saved_count += 1
                    next_part_num += 1
                   
        # Process and save Test
        if test_df is not None and len(test_df) > 0:
            flat_test = apply_prepare_data(test_df, seq_len=seq_len, skip_times=skip_times)
            if flat_test is not None and len(flat_test) > 0:
                test_dir = os.path.join(output_directory, "testing")
                os.makedirs(test_dir, exist_ok=True)
                chunks = split_dataframe_by_size(flat_test, MAX_CSV_SIZE, MAX_CSV_ROWS)
                logger.info("Test output split into %d chunk(s)", len(chunks))
               
                next_part_num = 1
                for chunk in chunks:
                    while True:
                        output_filename = f"prepared_samples_part{next_part_num:03d}.csv"
                        output_path = os.path.join(test_dir, output_filename)
                        if not os.path.exists(output_path):
                            break
                        next_part_num += 1
         
                    chunk.to_csv(output_path, index=False)
                    file_size = os.path.getsize(output_path)
                    logger.info("Saved %s in testing/ (%d rows, %.2f MB)",
                                output_filename, len(chunk), file_size / 1024 / 1024)
                    saved_count += 1
                    next_part_num += 1
 
        return saved_count, 0
 
def main():
    parser = argparse.ArgumentParser(
        description="Edge Impulse Transformation Block for CAN Data Anomaly Detection"
    )
    parser.add_argument('--bucket_name', type=str, required=False,
                        help="Bucket where your dataset is hosted")
    parser.add_argument('--window_size', type=int, default=30,
                        help="Window size in seconds for sample generation")
    parser.add_argument('--num_negative_samples', type=int, default=8,
                        help="Number of negative samples per VDP trip")
    parser.add_argument('--seq_len', type=int, default=30,
                        help="Number of rows to keep per sample before aggregation")
    parser.add_argument('--skip_times', type=int, default=10,
                        help="Number of trailing rows to skip per trip (leakage prevention)")
    parser.add_argument('--out-directory', type=str, required=False,
                        help="Output directory for generated files")
    parser.add_argument('--metadata', type=json.loads, required=False,
                        help="Existing metadata JSON")
 
    args, _ = parser.parse_known_args()
 
    if args.bucket_name:
        os.environ['BUCKET_NAME'] = args.bucket_name
 
    if not args.out_directory:
        args.out_directory = os.getenv('EI_OUTPUT_DIR', '/tmp/out')
    os.makedirs(args.out_directory, exist_ok=True)
 
    logger.info("=" * 60)
    logger.info("EDGE IMPULSE TRANSFORMATION BLOCK - CAN ANOMALY DETECTION")
    logger.info("=" * 60)
 
    metadata = {
        "version": 1,
        "action": "replace",
        "metadata": args.metadata if args.metadata else {}
    }
 
    input_base_path = detect_input_path(args)
    if not input_base_path:
        logger.error("No data source found.")
        metadata['metadata']['ei_prepared_samples'] = 0
        save_metadata(metadata, args.out_directory)
        return
 
    logger.info("Input base path: %s", input_base_path)
 
    pcs_files = get_files_from_specific_dirs(input_base_path, DATA_CONFIG["pcs_directories"])
    vdp_files = get_files_from_specific_dirs(input_base_path, DATA_CONFIG["vdp_directories"])
    logger.info("Found %d PCS files, %d VDP files", len(pcs_files), len(vdp_files))
 
    if not pcs_files and not vdp_files:
        logger.warning("No data files found in specified directories.")
        metadata['metadata']['ei_prepared_samples'] = 0
        save_metadata(metadata, args.out_directory)
        return
 
    num_success, num_failed = process_datasets(
        pcs_files, vdp_files,
        args.out_directory,
        window_size=args.window_size,
        num_negative_samples=args.num_negative_samples,
        seq_len=args.seq_len,
        skip_times=args.skip_times
    )
 
    logger.info("Output files generated: %d", num_success)
 
    metadata['metadata']['ei_prepared_samples'] = 1 if num_success > 0 else 0
    save_metadata(metadata, args.out_directory)
 
    logger.info("=" * 60)
    logger.info("TRANSFORMATION BLOCK COMPLETED")
    logger.info("=" * 60)
   
if __name__ == "__main__":
    main()