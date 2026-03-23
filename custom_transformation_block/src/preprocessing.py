import os
import sys

import pandas as pd
from tqdm import tqdm

# from src.utils.features import add_all_features
from .utils import file_utils

CONTINUOUS_AGG: dict[str, str] = {
    "VSC_GX0": "mean",
    "HV_ACCP": "mean",
    "VSC_YAW0": "mean",
    "PMC": "mean",
    "OTHLDIS": "mean",
    "SP1": "mean",
    "PWC": "mean",
    "SSA": "mean",
    "SSAV": "mean",
    "VSC_GY0": "mean",
}

CATEGORICAL_AGG: dict[str, str] = {
    "TNS": "first",
    "trip_id": "first",
    ########################################
    "B_P": "mean",
    "WSTP": "mean",
    "PKB_BDB": "mean",
    "LC": "mean",
}
RENAME_COLUMNS: dict[str, str] = {"GPS_TimeStamp": "timestamp"}
COLUMN_TO_CHECK_NAN: list[str] = [
    "VSC_GX0",
    "HV_ACCP",
    "VSC_YAW0",
    "PMC",
    "OTHLDIS",
    "SP1",
    "PWC",
    "SSA",
    "SSAV",
    "VSC_GY0",
    "TNS",
    "B_P",
    "WSTP",
    "PKB_BDB",
    "LC",
]


def create_trip_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique trip identifier."""
    df = df[(df["TRIP_COUNT"].notna()) & (df["Hased_VIN"].notna())]
    df["trip_id"] = df["Hased_VIN"] + "_" + df["TRIP_COUNT"].astype(int).astype(str)
    df = df.drop(["TRIP_COUNT", "Hased_VIN"], axis=1)
    return df


def resampling_data(trip_df: pd.DataFrame, target_rate_hz: int) -> pd.DataFrame:
    """
    Make sure data is uniformly sampled at the target_rate_hz.
    Process at ms level, default 20Hz
    Handles duplicate timestamps by adding a small offset to ensure uniqueness.
    """
    if trip_df is None or len(trip_df) == 0:
        return pd.DataFrame()
    # Make sure timestamp column is in datetime format
    trip_df["timestamp"] = pd.to_datetime(trip_df["timestamp"])
    trip_df = trip_df.sort_values(by=["trip_id", "timestamp"])

    # Set timestamp as index
    trip_df.set_index("timestamp", inplace=True)

    additional_columns = {}
    if "PCSBrakeAssistState" in trip_df.columns:
        additional_columns.update(
            {
                "PCSALM_count": "sum",
                "PCSBrakeAssistState": "max",
                "PCSALM": "max",
            }
        )
        trip_df["PCSALM_count"] = trip_df["PCSALM"].copy()
    # Resample to the target rate in ms using forward fill for missing data
    resampled_df = trip_df.resample(f"{int(1000 / target_rate_hz)}ms").agg(
        {**additional_columns, **CONTINUOUS_AGG, **CATEGORICAL_AGG}
    )

    return resampled_df


def preprocess_data(df: pd.DataFrame, output_path: str):
    print("Create trip id")
    df = create_trip_id(df)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    resampled_dfs = []
    for _, trip_df in tqdm(df.groupby("trip_id"), desc="Resampling trips"):
        resampled_trip_df = resampling_data(trip_df, target_rate_hz=1)
        resampled_dfs.append(resampled_trip_df)
    resampled_df = pd.concat(resampled_dfs)
    resampled_df.reset_index(inplace=True)
    # Drop rows with NaN values in COLUMN_TO_CHECK_NAN
    resampled_df = resampled_df.dropna(subset=COLUMN_TO_CHECK_NAN, how="all")
    # resampled_df = add_all_features(resampled_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if "PCSBrakeAssistState" in df.columns:
        print("Save to PCS file")
        resampled_df.reset_index().to_csv(output_path, index=False)
    else:
        print("Save to VDP file")
        resampled_df.reset_index().to_csv(output_path, index=False)

def process_dataframe_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core logic extracted from preprocess_data.
    Returns the processed dataframe instead of saving it fixing RAM problem.
    """
    print("Create trip id")
    df = create_trip_id(df)
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    resampled_dfs = []
    if "trip_id" in df.columns:
        for _, trip_df in tqdm(df.groupby("trip_id"), desc="Resampling trips"):
            resampled_trip_df = resampling_data(trip_df, target_rate_hz=1)
            resampled_dfs.append(resampled_trip_df)
    if not resampled_dfs:
        return pd.DataFrame()
 
    resampled_df = pd.concat(resampled_dfs)
    resampled_df.reset_index(inplace=True)
    cols_to_check = [c for c in COLUMN_TO_CHECK_NAN if c in resampled_df.columns]
    resampled_df = resampled_df.dropna(subset=cols_to_check, how="all")
    # resampled_df = add_all_features(resampled_df)
    # return df, do not save
    return resampled_df
def preprocess_data(df: pd.DataFrame, output_path: str):
    resampled_df = process_dataframe_logic(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if "PCSBrakeAssistState" in df.columns:
        print("Save to PCS file")
        resampled_df.reset_index().to_csv(output_path, index=False)
    else:
        print("Save to VDP file")
        resampled_df.reset_index().to_csv(output_path, index=False)

def prepare_data(data_name: str, output_path: str):
    if os.path.exists(output_path):
        return
    if data_name == "VDP":
        df = file_utils.get_current_raw_data()
        print("Starting preprocess VDP data")
        preprocess_data(df, output_path)
        print("VDP Data preprocessing completed.")
    elif data_name == "PCS":
        df = file_utils.get_current_pcs_raw_data()
        preprocess_data(df, output_path)
        print("PCS Data preprocessing completed.")
    else:
        raise ValueError(f"Unknown data name: {data_name}")
    print(f"Data {data_name} has been preprocessed and saved to {output_path}")


