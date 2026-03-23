"""
Generate training samples for anomaly detection WITHOUT train/test/val split.
Creates a single unified dataset from all available trips.
 
B2 mode:
- Preprocessing exports BASE table (no feature engineering).
- This generator loads BASE columns, creates `times` from `timestamp`,
  then calls `add_all_features()` to decorate features for sampling.
"""
# Standard library imports
from asyncio.log import logger
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
from sklearn.model_selection import StratifiedShuffleSplit
 
# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
 
# Local application imports
from src.preprocessing import prepare_data
from src.utils.features import add_all_features
 
 
class AnomalyDataGenerator:
    """Generate anomaly detection samples without splitting."""
 
    BASE_COLUMNS: List[str] = [
        "index",
        "timestamp",
        "trip_id",
        "B_P",
        "HV_ACCP",
        "OTHLDIS",
        "PKB_BDB",
        "PMC",
        "SP1",
        "SSA",
        "SSAV",
        "TNS",
        "VSC_GX0",
        "VSC_GY0",
        "VSC_YAW0",
        "WSTP",
        "PWC",
        "LC",
    ]
 
    def __init__(
        self,
        pcs_data_path: str,
        gen_data_path: str,
        output_dir: str = "./data/input",
        window_size: int = 30,
        skip_window_size: int = 10,
        test_ratio: float = 0.2,
        random_state: int = 60,
        num_negative_samples: int = 8,
    ):
        self.pcs_data_path = pcs_data_path
        self.gen_data_path = gen_data_path
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.skip_window_size = skip_window_size
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.num_negative_samples = num_negative_samples
 
        logger.info(f"Num negative samples: {num_negative_samples}")
        self.pcs_data: Optional[pd.DataFrame] = None
        self.gen_data: Optional[pd.DataFrame] = None
        self.pcs_dict: Dict[str, pd.DataFrame] = {}
        self.gen_dict: Dict[str, pd.DataFrame] = {}
 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_seeds(random_state)
 
    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
 
    def _ensure_base_columns(self, df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df
 
    def _make_times_from_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
 
        df["times"] = (
            df.groupby("trip_id")["timestamp"]
            .transform(lambda s: (s - s.min()).dt.total_seconds())
            .astype(np.float64)
        )
        return df
 
    def load_data(self) -> None:
        pcs_cols = [c for c in self.BASE_COLUMNS if c != "index"] + ["index"]
        gen_cols = [c for c in self.BASE_COLUMNS if c != "index"] + ["index"]
       
        # Load PCS
        if self.pcs_data_path and os.path.exists(self.pcs_data_path) and os.path.getsize(self.pcs_data_path) > 0:
            print("Loading PCS data...")
           
 
            self.pcs_data = pd.read_csv(self.pcs_data_path, usecols=pcs_cols + ["PCSALM"])
            self.pcs_data.rename(columns={"PCSALM": "truth_label"}, inplace=True)
            # Create a trip id feature for prevent data leakage
            self.pcs_data["group_trip_id"] = self.pcs_data["trip_id"].astype(str)
 
            self.pcs_data["trip_id"] = self.pcs_data["group_trip_id"] + "_pcs"
 
 
            self.pcs_data = self._make_times_from_timestamp(self.pcs_data)
            self.pcs_data = self.pcs_data.fillna(0)
            self.pcs_data = add_all_features(self.pcs_data)
            pcs_with_alarms = self.pcs_data[self.pcs_data["truth_label"] == 1]["trip_id"].unique()
            self.pcs_data = self.pcs_data[self.pcs_data["trip_id"].isin(pcs_with_alarms)].copy()
        else:
            print("PCS data path is empty or missing. Skipping PCS load.")
            self.pcs_data = pd.DataFrame()
 
        # Load GEN
        if self.gen_data_path and os.path.exists(self.gen_data_path) and os.path.getsize(self.gen_data_path) > 0:
            print("Loading generated data...")
 
            self.gen_data = pd.read_csv(self.gen_data_path, usecols=gen_cols)
            # Create a trip id feature for prevent data leakage
            self.gen_data["group_trip_id"] = self.gen_data["trip_id"].astype(str)
            self.gen_data["trip_id"] = self.gen_data["group_trip_id"] + "_gen"
 
            self.gen_data = self._make_times_from_timestamp(self.gen_data)
            self.gen_data = self.gen_data.fillna(0)
            self.gen_data = add_all_features(self.gen_data)
        else:
            print("Generated data path is empty or missing. Skipping GEN load.")
            self.gen_data = pd.DataFrame()
 
        print(f"PCS data: {len(self.pcs_data)} rows")
        print(f"Generated data: {len(self.gen_data)} rows")
 
    def dump_raw_data(self) -> None:
        if not self.pcs_data.empty:
            self.pcs_data.to_csv(self.output_dir / "pcs_data.csv", index=False)
        if not self.gen_data.empty:
            self.gen_data.to_csv(self.output_dir / "gen_data.csv", index=False)
 
    def build_trip_dictionaries(self) -> None:
        if not self.pcs_data.empty:
            self.pcs_data = self.pcs_data.sort_values(["trip_id", "times"]).reset_index(drop=True)
            self.pcs_dict = {
                trip_id: group.reset_index(drop=True)
                for trip_id, group in self.pcs_data.groupby("trip_id", sort=True)
            }
       
        if not self.gen_data.empty:
            self.gen_data = self.gen_data.sort_values(["trip_id", "times"]).reset_index(drop=True)
            self.gen_dict = {
                trip_id: group.reset_index(drop=True)
                for trip_id, group in self.gen_data.groupby("trip_id", sort=True)
            }
 
 
    def downsample_gen_trips_to_match_pcs(
        self,
        seed: int = 42,
    ) -> None:
 
        pcs_trip_ids = sorted(list(self.pcs_dict.keys()))
        gen_trip_ids = sorted(list(self.gen_dict.keys()))
        rng = np.random.RandomState(seed)
        sampled_indices = rng.choice(len(gen_trip_ids), size=len(pcs_trip_ids), replace=False)
        sampled_indices_sorted = np.sort(sampled_indices)
        gen_trip_ids_sampled = [gen_trip_ids[i] for i in sampled_indices_sorted]
 
        selected_set = set(gen_trip_ids_sampled)
 
        self.gen_dict = {tid: df for tid, df in self.gen_dict.items() if tid in selected_set}
 
 
    def extract_window_sample(
        self,
        trip_df: pd.DataFrame,
        point_idx: int,
        trip_id: str,
        label: int,
        suffix: str = "",
    ) -> Optional[pd.DataFrame]:
        start_idx = max(0, point_idx - self.window_size - self.skip_window_size)
        if point_idx - start_idx < self.window_size:
            return None
 
        window_data = trip_df.iloc[start_idx : point_idx + 1].copy()
        window_data["new_trip_id"] = f"{trip_id}{suffix}_{start_idx}_{point_idx}"
        window_data["label"] = label
        return window_data
 
    def create_positive_samples(self, trip_id: str, pcs_trip: pd.DataFrame) -> List[pd.DataFrame]:
        samples: List[pd.DataFrame] = []
        try:
            pcs_trip = pcs_trip.reset_index(drop=True)
            if "truth_label" not in pcs_trip.columns:
                return samples
 
            truth_labels = pcs_trip["truth_label"].values
            first_alarm_idx = np.argmax(truth_labels == 1)
            if truth_labels[first_alarm_idx] != 1:
                return samples
 
            window = self.extract_window_sample(
                pcs_trip, first_alarm_idx, trip_id, label=1, suffix=""
            )
            if window is None:
                return samples
 
            samples.append(window)
        except (KeyError, IndexError, ValueError):
            return samples
        return samples
 
    def create_negative_samples(
        self, trip_id: str, gen_trip: pd.DataFrame, num_samples: int = 10
    ) -> List[pd.DataFrame]:
        samples: List[pd.DataFrame] = []
        try:
            gen_len = len(gen_trip)
            min_idx = self.window_size + 1
            if gen_len < min_idx:
                return samples
 
            available_range = gen_len - min_idx
            max_samples = available_range // (self.window_size + 1)
            if max_samples <= 0:
                return samples
 
            trip_hash = hashlib.md5(str(trip_id).encode()).hexdigest()
            trip_seed = int(trip_hash, 16) % (2**32)
            trip_rng = np.random.RandomState(trip_seed)
 
            selected_indices = []
            available_indices = list(range(min_idx, gen_len))
            for _ in range(max_samples):
                if not available_indices:
                    break
                idx_position = trip_rng.randint(0, len(available_indices))
                idx = available_indices[idx_position]
                selected_indices.append(idx)
 
                overlap_start = max(min_idx, idx - self.window_size - 1)
                overlap_end = min(gen_len, idx + self.window_size + 2)
                available_indices = [
                    i for i in available_indices if i < overlap_start or i >= overlap_end
                ]
 
            random.seed(trip_seed)
            random.shuffle(selected_indices)
 
            created_count = 0
            for idx in selected_indices:
                if created_count >= num_samples:
                    break
                window = self.extract_window_sample(
                    gen_trip, idx, trip_id, label=0, suffix=""
                )
                if window is not None:
                    samples.append(window)
                    created_count += 1
        except (KeyError, IndexError, ValueError):
            return samples
        return samples
 
    def split_train_test(self) -> Tuple[List[str], List[str]]:
        """Split trip IDs into train and test sets using StratifiedShuffleSplit"""
        pcs_trip_ids = sorted(list(self.pcs_dict.keys()))
        gen_trip_ids = sorted(list(self.gen_dict.keys()))
 
        all_trip_ids = pcs_trip_ids + gen_trip_ids
        df = pd.DataFrame({
            "trip_id": all_trip_ids,
            "source": [1] * len(pcs_trip_ids) + [0] * len(gen_trip_ids)
        })
 
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
        train_idx, test_idx = next(sss.split(df, df["source"]))
 
        train_trip_ids = df.iloc[train_idx]["trip_id"].tolist()
        test_trip_ids = df.iloc[test_idx]["trip_id"].tolist()
 
        logger.info(f"Train trips: {len(train_trip_ids)} | Test trips: {len(test_trip_ids)}")
        return train_trip_ids, test_trip_ids
 
    def generate_all_samples(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        train_trip_ids, test_trip_ids = self.split_train_test()
       
        train_samples: List[pd.DataFrame] = []
        test_samples: List[pd.DataFrame] = []
 
        pcs_trip_ids = sorted(list(self.pcs_dict.keys()))
        gen_trip_ids = sorted(list(self.gen_dict.keys()))
 
        for trip_id in tqdm(pcs_trip_ids, desc="Processing PCS trips"):
            samples = self.create_positive_samples(trip_id, self.pcs_dict[trip_id])
            if trip_id in train_trip_ids:
                train_samples.extend(samples)
            else:
                test_samples.extend(samples)
 
        for trip_id in tqdm(gen_trip_ids, desc="Processing GEN trips"):
            samples = self.create_negative_samples(
                trip_id, self.gen_dict[trip_id], self.num_negative_samples
            )
            if trip_id in train_trip_ids:
                train_samples.extend(samples)
            else:
                test_samples.extend(samples)
 
        return train_samples, test_samples
 
    def dump_samples(self, train_samples: List[pd.DataFrame], test_samples: List[pd.DataFrame]) -> None:
        if train_samples:
            pd.concat(train_samples, ignore_index=True).to_csv(self.output_dir / "train_samples.csv", index=False)
        if test_samples:
            pd.concat(test_samples, ignore_index=True).to_csv(self.output_dir / "test_samples.csv", index=False)
 
    def run(self) -> None:
        self.load_data()
        self.dump_raw_data()
        self.build_trip_dictionaries()
       
        self.downsample_gen_trips_to_match_pcs(seed=60)
 
        train_samples, test_samples = self.generate_all_samples()
        self.dump_samples(train_samples, test_samples)
       
# if __name__ == "__main__":
#     PCS_DATA_PATH = "/home/anhtnt18/Downloads/pcs1_d2ata.csv"
#     GEN_DATA_PATH = "/home/anhtnt18/Downloads/vdp1_dat3a.csv"
#     OUTPUT_DIR = "./data/output"
 
#     prepare_data("PCS", PCS_DATA_PATH)
#     prepare_data("VDP", GEN_DATA_PATH)
 
#     generator = AnomalyDataGenerator(
#         pcs_data_path=PCS_DATA_PATH,
#         gen_data_path=GEN_DATA_PATH,
#         output_dir=OUTPUT_DIR,
#         window_size=30,
#         skip_window_size=10,
#         random_state=60,
#         num_negative_samples=8,
#     )
#     generator.run()