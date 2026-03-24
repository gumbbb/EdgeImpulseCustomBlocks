import numpy as np
import pandas as pd
from src.utils.features import add_all_features
from src.utils.model_utils import prepare_data
from sklearn.preprocessing import StandardScaler

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, **kwargs):
    # Process the raw signal
    num_axes = len(axes)
    num_rows = len(raw_data) // num_axes
    reshaped_data = np.array(raw_data).reshape((num_rows, num_axes))
    
    df = pd.DataFrame(reshaped_data, columns=axes)
    
    # --- PREPARE FOR ORIGINAL CODE PARITY ---
    # The original native functions require certain identifying columns
    if 'trip_id' not in df.columns:
        df['trip_id'] = 'sample_window'
    if 'new_trip_id' not in df.columns:
        df['new_trip_id'] = 'sample_window'
    if 'group_trip_id' not in df.columns:
        df['group_trip_id'] = 'sample_window'
    if 'label' not in df.columns:
        df['label'] = 0  # Required for prepare_data max() grouping if missing
        
    # features.py requires 'timestamp' to parse into standard pd.datetime properly
    if 'timestamp' in df.columns:
        # Edge impulse sends purely integer ms timestamps on time-series blocks
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').fillna(pd.to_datetime('2025-01-01'))
    else:
        df['timestamp'] = pd.to_datetime(np.arange(len(df)), unit='s', origin='2025-01-01')
    
    # --- STEP 4: FEATURE ENGINEERING ---
    # Exactly utilizing the original logic
    df = add_all_features(df, groupby_column='trip_id', time_column='timestamp')
    
    # Cleaning identical to load_data()
    df = df.fillna(0)
    if 'TNS' in df.columns:
        df['TNS'] = df['TNS'].astype(int)
        
    # --- STEP 4: AGGREGATE ---
    # Using the exact original `prepare_data` from model_utils.py.
    # This automatically invokes .tail(seq_len) and groups by mean/max!
    X, y, selected_cols, meta = prepare_data(df, seq_len=30)

    # --- STEP 5: Data Normalizing (Scaling) ---
    # Apply standard normalizer
    scaler = StandardScaler()
    # Safely apply to prevent zeroing-out if Edge Impulse only passes 1 sample batch in `/run`
    if X.shape[0] > 1:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X # If 1 sample, variance is 0. Cannot fit standard scalar across a point.
        
    features = X_scaled.flatten().tolist()
    
    result = {
        'features': features,
        'graphs': [],
        'output_config': {
            'type': 'flat',
            'shape': {
                'width': len(features)
            }
        }
    }
    
    if y is not None and len(y) > 0:
        result['labels'] = [str(int(y[0]))]
        
    return result
