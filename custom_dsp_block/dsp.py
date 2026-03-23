import numpy as np
import pandas as pd
from src.utils.features import add_all_features
from sklearn.preprocessing import StandardScaler

LGBM_FEATURES = [
    "B_P", "OTHLDIS", "PKB_BDB", "SP1", "SSA", "times",
    "avg_sudden_acceleration_count", "avg_harsh_break_count",
    "avg_speed", "lc_count", "max_speed_continuous_above_130",
    "TTC_filled", "lane_change_behavior_flag", "unsteady_driving_flag"
]

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq, **kwargs):
    # Process the raw signal
    num_axes = len(axes)
    num_rows = len(raw_data) // num_axes
    reshaped_data = np.array(raw_data).reshape((num_rows, num_axes))
    
    df = pd.DataFrame(reshaped_data, columns=axes)
    
    # Feature Engineering
    if 'timestamp' not in df.columns and 'times' not in df.columns:
        df['times'] = np.arange(len(df))
        
    df = add_all_features(df)
    df = df.fillna(0)
    
    if 'TNS' in df.columns:
        df['TNS'] = df['TNS'].astype(int)
        
    cols_to_select = LGBM_FEATURES.copy()
    for col in cols_to_select:
        if col not in df.columns:
            df[col] = 0.0
            
    df_selected = df[cols_to_select].copy()
    
    agg_dict = {col: 'mean' for col in cols_to_select}
    if 'label' in df.columns:
        agg_dict['label'] = 'max'
        df_selected['label'] = df['label']
        
    aggregated = df_selected.agg(agg_dict).to_frame().T
    X = aggregated[LGBM_FEATURES].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    features = X_scaled.flatten().tolist()
    
    determined_label = None
    if 'label' in aggregated.columns:
        determined_label = str(int(aggregated['label'].iloc[0]))
        
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
    
    if determined_label is not None:
        result['labels'] = [determined_label]
        
    return result
