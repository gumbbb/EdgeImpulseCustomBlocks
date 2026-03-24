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
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import tempfile

from src.preprocessing import preprocess_data
from src.gen_data import AnomalyDataGenerator
import src.gen_data

# **CRITICAL MONKEYPATCH**: We explicitly DO NOT want the Transformation block 
# computing DSP features. We override `add_all_features` to return the df unmodified.
src.gen_data.add_all_features = lambda df: df 

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

def detect_input_path(args) -> Optional[str]:
    """Detect input data path using exact original tiered fallback strategy."""
    input_base_path = None

    # Tier 1: Argument-based
    if args.bucket_name:
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
                subdirs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
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

def preprocess_and_combine_files(csv_files: List[str], temp_dir: str, data_type: str) -> Optional[str]:
    """Resample combinations to 1Hz using src.preprocessing."""
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
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensures truth labels carry over correctly into the 1Hz sample logic for PCS
    if data_type == 'PCS' and "PCSALM" not in combined_df.columns and "PCSBrakeAssistState" in combined_df.columns:
         combined_df["PCSALM"] = combined_df["PCSBrakeAssistState"]

    output_path = os.path.join(temp_dir, f"{data_type.lower()}_preprocessed.csv")
    try:
        preprocess_data(combined_df, output_path)
        logger.info("Preprocessing complete for %s -> %s", data_type, output_path)
    except Exception as e:
        logger.error("Preprocessing failed for %s: %s", data_type, e, exc_info=True)
        return None

    return output_path

def save_window_as_ei_csv(window_df: pd.DataFrame, out_dir: str, file_prefix: str, label: int):
    """
    Apply The 'Cut' (leakage prevention) if needed, format the timestamp correctly, 
    and save exactly 1 window per CSV to the Edge Impulse output directory.
    """
    # Determine category for subdirectory (training vs testing)
    category_dir = "testing" if "test" in file_prefix else "training"
    save_dir = os.path.join(out_dir, category_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Must format for EI time-series: first column is timestamp (ms)
    df_out = window_df.copy()
    
    # Fix timestamp logic: if concatenating, we must ensure timestamps are continuous 
    # and not resetting for every window in the same file.
    # For a single file, we use a global relative timestamp.
    df_out['timestamp'] = (np.arange(len(df_out)) * 1000).astype(int)
        
    # Ensure the label column is present for the CSV Wizard
    df_out['label'] = label

    # Rearrange and drop internal processing columns
    processed_cols = ['index', 'trip_id', 'new_trip_id', 'truth_label', 'times', 'group_trip_id', 'PCSALM']
    cols = ['timestamp'] + [c for c in df_out.columns if c not in processed_cols and c != 'timestamp']
    
    # Ensure stable ordering for Edge Impulse
    df_out = df_out[cols]

    filepath = os.path.join(save_dir, f"{file_prefix}.csv")
    df_out.to_csv(filepath, index=False)

def process_datasets(pcs_files: List[str], vdp_files: List[str],
                     output_directory: str,
                     window_size: int = 30,
                     num_negative_samples: int = 8,
                     skip_times: int = 10) -> int:
    """Core logic: Resample -> Downsample & Window -> Apply Cut -> Save CSVs"""
    saved_count = 0
    with tempfile.TemporaryDirectory(prefix="ei_transform_") as temp_dir:
        pcs_path = preprocess_and_combine_files(pcs_files, temp_dir, 'PCS')
        vdp_path = preprocess_and_combine_files(vdp_files, temp_dir, 'VDP')

        if not pcs_path and not vdp_path:
            return 0

        generator = AnomalyDataGenerator(
            pcs_data_path=pcs_path,
            gen_data_path=vdp_path,
            output_dir=temp_dir,
            window_size=window_size,
            skip_window_size=10, # Hardcoded in original script
            random_state=42,
            num_negative_samples=num_negative_samples
        )

        generator.load_data()
        generator.build_trip_dictionaries()
        generator.downsample_gen_trips_to_match_pcs(seed=42)
        train_samples, test_samples = generator.generate_all_samples()
        
        logger.info("Concatenating %d train and %d test windows into single dataframes...", len(train_samples), len(test_samples))
        
        final_train_dfs = []
        for i, window_df in enumerate(train_samples):
            cut_df = window_df.iloc[: -(skip_times + 1)].copy()
            if len(cut_df) > 0:
                # Add explicit label column before concatenation
                cut_df['label'] = window_df['label'].iloc[0]
                final_train_dfs.append(cut_df)

        final_test_dfs = []
        for i, window_df in enumerate(test_samples):
            cut_df = window_df.iloc[: -(skip_times + 1)].copy()
            if len(cut_df) > 0:
                cut_df['label'] = window_df['label'].iloc[0]
                final_test_dfs.append(cut_df)

        # Save single unified training file
        if final_train_dfs:
            train_big_df = pd.concat(final_train_dfs, ignore_index=True)
            save_window_as_ei_csv(train_big_df, output_directory, "training_data", label=0) # label value is dummy here as it's inside DF
            saved_count += 1

        # Save single unified testing file
        if final_test_dfs:
            test_big_df = pd.concat(final_test_dfs, ignore_index=True)
            save_window_as_ei_csv(test_big_df, output_directory, "testing_data", label=0)
            saved_count += 1

    return saved_count

    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Edge Impulse Transformation Block - Standalone Mode")
    parser.add_argument('--bucket_name', type=str, required=False, help="Bucket hosted directory")
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--num_negative_samples', type=int, default=8)
    parser.add_argument('--skip_times', type=int, default=10, help="Leakage prevention cut")
    parser.add_argument('--out-directory', type=str, required=False, help="Output output for EI CSVs")
    parser.add_argument('--metadata', type=str, required=False, help="Existing metadata block") # Pass as str, parse later if needed
    args, _ = parser.parse_known_args()

    if not args.out_directory:
        args.out_directory = os.getenv('EI_OUTPUT_DIR', '/tmp/out')
    os.makedirs(args.out_directory, exist_ok=True)

    # Ensure metadata is parsed securely if provided as a string via CLI
    metadata_val = {}
    if args.metadata:
        try:
            metadata_val = json.loads(args.metadata)
        except Exception as e:
            logger.warning("Failed to parse metadata string: %s", e)
            
    metadata = {
        "version": 1,
        "action": "replace",
        "metadata": metadata_val
    }

    input_base_path = detect_input_path(args)
    if not input_base_path:
        logger.error("No input data path detected.")
        metadata['metadata']['ei_prepared_samples'] = 0
        save_metadata(metadata, args.out_directory)
        return

    pcs_files = get_files_from_specific_dirs(input_base_path, DATA_CONFIG["pcs_directories"])
    vdp_files = get_files_from_specific_dirs(input_base_path, DATA_CONFIG["vdp_directories"])

    num_success = process_datasets(
        pcs_files, vdp_files,
        args.out_directory,
        window_size=args.window_size,
        num_negative_samples=args.num_negative_samples,
        skip_times=args.skip_times
    )

    metadata['metadata']['ei_prepared_samples'] = 1 if num_success > 0 else 0
    save_metadata(metadata, args.out_directory)
    logger.info("Transformation Block completed. Saved %d samples.", num_success)

if __name__ == "__main__":
    main()
