import numpy as np
import pandas as pd


def add_relative_time(
    df: pd.DataFrame, time_column: str = "timestamp", groupby_column: str = "trip_id", new_column_name: str = "times"
) -> pd.DataFrame:
    """
    Add relative_time column (in seconds) for each trip from the start of the trip.
    """
    result = []
    df[time_column] = pd.to_datetime(df[time_column])
    for _, trip_df in df.groupby(by=groupby_column):
        trip_df = trip_df.copy()  # Avoid SettingWithCopyWarning
        start_time = trip_df[time_column].min()
        trip_df[new_column_name] = (trip_df[time_column] - start_time).dt.total_seconds()
        result.append(trip_df)
    return pd.concat(result, ignore_index=True)


def add_avg_sudden_acceleration_count(
    df: pd.DataFrame,
    window_size: int = 5,
    gx0_threshold: float = 0.31,
    groupby_column: str = "trip_id",
    new_column_name: str = "avg_sudden_acceleration_count",
) -> pd.DataFrame:
    """
    Add avg_sudden_acceleration_count column which counts the number of sudden accelerations
    (above a certain threshold) in a rolling window and computes the average per trip.
    """
    result = []
    for _, trip_df in df.groupby(by=groupby_column):
        trip_df = trip_df.copy()  # Avoid SettingWithCopyWarning
        trip_df[new_column_name] = (trip_df["VSC_GX0"] > gx0_threshold).astype(int).rolling(window=window_size).mean()
        result.append(trip_df)
    return pd.concat(result, ignore_index=True)


def add_avg_harsh_break_count(
    df: pd.DataFrame,
    window_size: int = 3,
    gx0_threshold: float = 0.31,
    groupby_column: str = "trip_id",
    new_column_name: str = "avg_harsh_break_count",
) -> pd.DataFrame:
    """
    Add avg_sudden_acceleration_count column which counts the number of sudden accelerations
    (above a certain threshold) in a rolling window and computes the average per trip.
    """
    result = []
    for _, trip_df in df.groupby(by=groupby_column):
        trip_df = trip_df.copy()  # Avoid SettingWithCopyWarning
        trip_df[new_column_name] = (trip_df["VSC_GX0"] < -gx0_threshold).astype(int).rolling(window=window_size).mean()
        result.append(trip_df)
    return pd.concat(result, ignore_index=True)


def add_avg_speed(
    df: pd.DataFrame, window_size: int = 5, groupby_column: str = "trip_id", new_column_name: str = "avg_speed"
) -> pd.DataFrame:
    result = []
    for _, trip_df in df.groupby(by=groupby_column):
        trip_df = trip_df.copy()  # Avoid SettingWithCopyWarning
        trip_df[new_column_name] = trip_df["SP1"].rolling(window=window_size).mean()
        result.append(trip_df)
    return pd.concat(result, ignore_index=True)


def add_lc_count(
    df: pd.DataFrame, window_size: int = 5, groupby_column: str = "trip_id", new_column_name: str = "lc_count"
) -> pd.DataFrame:
    """
    Add lc_count column which counts the number of lane changes (LC == 1)
    in a rolling window.
    """
    result = []
    for _, trip_df in df.groupby(by=groupby_column):
        trip_df = trip_df.copy()  # Avoid SettingWithCopyWarning
        trip_df[new_column_name] = (trip_df["LC"] > 0).astype(int).rolling(window=window_size).sum()
        result.append(trip_df)
    return pd.concat(result, ignore_index=True)


def add_max_continuous_above_130(
    df: pd.DataFrame,
    speed_col: str = "SP1",
    trip_col: str = "trip_id",
    threshold_kmh: float = 130.0,
    out_col: str = "max_speed_continuous_above_130",
) -> pd.DataFrame:
    """
    130km/h 以上が何秒続いたか（1秒刻み前提）をカウントして out_col に入れる簡易版。
    """
    if speed_col not in df.columns or trip_col not in df.columns:
        raise KeyError(f"{speed_col} または {trip_col} 列がありません。")

    df = df.copy()
    df["_high_speed_flag"] = (df[speed_col].astype(float) >= threshold_kmh).astype(int)
    df[out_col] = 0

    # trip ごとに True の連続長を数える
    for _, sub in df.groupby(trip_col, sort=False):
        run = 0
        for i in sub.index:
            if df.at[i, "_high_speed_flag"] == 1:
                run += 1
            else:
                run = 0
            df.at[i, out_col] = run

    df = df.drop(columns=["_high_speed_flag"])
    return df


def add_ttc(df: pd.DataFrame, max_othldis: int = 125) -> pd.DataFrame:
    """
    Calculate TTC (Time To Collision) and fill NaN values.
    Uses pandas only for better performance.
    """
    df = df.copy()

    # 1. Calculate relative_speed (change in distance between frames)
    df["relative_speed"] = df.groupby("trip_id")["OTHLDIS"].diff()
    # Set relative_speed to NaN where distance is >= 125
    df.loc[df["OTHLDIS"] >= max_othldis, "relative_speed"] = np.nan

    # 2. Calculate TTC where relative_speed is negative (approaching)
    df["TTC"] = np.where(
        (df["relative_speed"].notna()) & (df["relative_speed"] < 0), df["OTHLDIS"] / (-df["relative_speed"]), np.nan
    )

    # 3. Fill NaN values in TTC
    df["TTC_filled"] = df["TTC"].fillna(-1)

    # 4. Create flag for original NaN values
    df["flag_dm import tqdTTC_null"] = df["TTC"].isna().astype(np.int8)

    return df


def add_lane_change_behavior_flag_simple(
    df: pd.DataFrame,
    speed_col: str = "SP1",
    yaw_rate_col: str = "VSC_YAW0",
    tns_col: str = "TNS",
    sampling_rate_hz: float = 1.0,
    horizon_sec: float = 3.0,
    window_sec: float = 3.0,
    no_signal_value: float = 3.0,
    min_lateral_m: float = 10.0,
    min_angle_deg: float = 20.0,
    lateral_col: str = "lateral_displacement",
    angle_col: str = "angle_change",
    out_col: str = "lane_change_behavior_flag",
) -> pd.DataFrame:
    """
    合図なし進路変更フラグ。
    必要に応じて lateral_displacement と angle_change を内部で計算。
    """
    for c in [speed_col, yaw_rate_col, tns_col]:
        if c not in df.columns:
            raise KeyError(f"{c} 列がありません。")

    df = df.copy()

    # 1) 横移動量のごく簡易な近似
    if lateral_col not in df.columns:
        speed_mps = df[speed_col].astype(float) * (1000.0 / 3600.0)
        yaw = df[yaw_rate_col].astype(float)
        # v * |yaw| * t^2 / 2 というかなり単純な近似
        df[lateral_col] = speed_mps * np.abs(yaw) * (horizon_sec**2) / 2.0

    # 2) angle_change（ローリング窓内の max-min）
    if angle_col not in df.columns:
        window_size = max(int(round(window_sec * sampling_rate_hz)), 1)
        df[angle_col] = (
            df[yaw_rate_col]
            .astype(float)
            .rolling(window=window_size, min_periods=1)
            .apply(lambda x: x.max() - x.min(), raw=True)
        )

    # 3) 条件判定
    cond = (df[tns_col] == no_signal_value) & (df[lateral_col] >= min_lateral_m) & (df[angle_col] >= min_angle_deg)
    df[out_col] = cond.astype(int)
    return df


# unsteady_driving_flag
def add_unsteady_driving_flag(
    df: pd.DataFrame,
    speed_col: str = "SP1",
    yaw_rate_col: str = "VSC_YAW0",
    sampling_rate_hz: float = 1.0,
    horizon_sec: float = 3.0,
    window_sec: float = 10.0,
    max_lateral_m: float = 5.0,
    min_angle_deg: float = 30.0,
    lateral_col: str = "lateral_displacement",
    angle_col: str = "angle_change_unsteady",
    out_col: str = "unsteady_driving_flag",
) -> pd.DataFrame:
    """
    ふらつき運転フラグ。
    - 横移動量は小さい
    - 角度変化（長めの窓）は大きい
    """
    for c in [speed_col, yaw_rate_col]:
        if c not in df.columns:
            raise KeyError(f"{c} 列がありません。")

    df = df.copy()

    # 1) 横移動量（lane_change 用と同じ近似。なければ計算）
    if lateral_col not in df.columns:
        speed_mps = df[speed_col].astype(float) * (1000.0 / 3600.0)
        yaw = df[yaw_rate_col].astype(float)
        df[lateral_col] = speed_mps * np.abs(yaw) * (horizon_sec**2) / 2.0

    # 2) 長めの窓での angle_change
    if angle_col not in df.columns:
        window_size = max(int(round(window_sec * sampling_rate_hz)), 1)
        df[angle_col] = (
            df[yaw_rate_col]
            .astype(float)
            .rolling(window=window_size, min_periods=1)
            .apply(lambda x: x.max() - x.min(), raw=True)
        )

    cond = (df[lateral_col] <= max_lateral_m) & (df[angle_col] >= min_angle_deg)
    df[out_col] = cond.astype(int)
    return df


def add_all_features(df: pd.DataFrame, groupby_column: str = "trip_id", time_column: str = "timestamp") -> pd.DataFrame:
    """
    Run all feature engineering functions.
    """
    org_cols = df.columns
    df.sort_values(by=[groupby_column, time_column], inplace=True)
    df = add_relative_time(df, groupby_column=groupby_column).sort_values(by=[groupby_column, time_column])
    df = add_avg_sudden_acceleration_count(
        df, window_size=5, gx0_threshold=0.31, groupby_column=groupby_column
    ).sort_values(by=[groupby_column, time_column])
    df = add_avg_harsh_break_count(df, window_size=3, gx0_threshold=0.31, groupby_column=groupby_column).sort_values(
        by=[groupby_column, time_column]
    )
    df = add_avg_speed(df, window_size=5, groupby_column=groupby_column).sort_values(by=[groupby_column, time_column])
    df = add_lc_count(df, window_size=5, groupby_column=groupby_column).sort_values(by=[groupby_column, time_column])

    df = add_max_continuous_above_130(
        df, speed_col="SP1", trip_col=groupby_column, threshold_kmh=130.0, out_col="max_speed_continuous_above_130"
    ).sort_values(by=[groupby_column, time_column])
    df = add_ttc(df, max_othldis=125).sort_values(by=[groupby_column, time_column])
    df = add_lane_change_behavior_flag_simple(df).sort_values(by=[groupby_column, time_column])
    df = add_unsteady_driving_flag(df).sort_values(by=[groupby_column, time_column])
    # added columns
    print("Added columns:", df.columns.difference(org_cols))
    return df
