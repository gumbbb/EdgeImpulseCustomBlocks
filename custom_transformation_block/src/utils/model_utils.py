"""
Shared utility functions for all ML models and experiments.
Used across exp000_iTransformer, exp001_baseline_lstm, exp002_lr, exp004_xgb, etc.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc,
    matthews_corrcoef, precision_recall_curve
)
from collections import Counter
 
try:
    from torch.utils.data import WeightedRandomSampler
except ImportError:
    WeightedRandomSampler = None
 
 
def load_data(train_path: str, val_path: str, test_path: str, skip_times: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, validation and test datasets from parquet files."""
    train_df = pd.read_parquet(train_path)
    val_df = None
    test_df = None
   
    if val_path:
        val_df = pd.read_parquet(val_path)
   
    if test_path:
        test_df = pd.read_parquet(test_path)
 
    skip_n_last_rows = skip_times + 1  # e.g., skip last 4 rows
    # Remove last 4 rows of each trip in train, val, test sets
    def remove_last_n_rows(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
        """Remove last n rows from each group after sorting by 'times'."""
        df = df.sort_values(['new_trip_id', 'times'])
        indices_to_keep = df.groupby('new_trip_id', sort=False).cumcount() < (df.groupby('new_trip_id', sort=False).transform('size') - n)
        return df[indices_to_keep].reset_index(drop=True)
   
    train_df = remove_last_n_rows(train_df, skip_n_last_rows)
    if val_df is not None:
        val_df = remove_last_n_rows(val_df, skip_n_last_rows)
    if test_df is not None:
        test_df = remove_last_n_rows(test_df, skip_n_last_rows)
 
    # fill nans by zeros
    train_df = train_df.fillna(0)
    if val_df is not None:
        val_df = val_df.fillna(0)
    if test_df is not None:
        test_df = test_df.fillna(0)
   
    # convert float to int in TNS column
    train_df["TNS"] = train_df["TNS"].astype(int)
    if val_df is not None:
        val_df["TNS"] = val_df["TNS"].astype(int)
    if test_df is not None:
        test_df["TNS"] = test_df["TNS"].astype(int)
   
    return train_df, val_df, test_df
 
 
def _encode_tns_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Helper function to one-hot encode TNS categorical column.
   
    Args:
        df: DataFrame containing TNS column
   
    Returns:
        Tuple of (tns_dummies_df, tns_column_names_list)
    """
    # One-hot encode TNS categorical column (values 0, 1, 2, 3)
    tns_dummies = pd.get_dummies(df['TNS'], prefix='TNS', dtype=int)
   
    # Ensure all 4 categories exist (0, 1, 2, 3)
    tns_cols = [f'TNS_{i}' for i in range(4)]
    for col_name in tns_cols:
        if col_name not in tns_dummies.columns:
            tns_dummies[col_name] = 0
   
    # Reorder columns to ensure consistent ordering
    tns_dummies = tns_dummies[tns_cols]
   
    return tns_dummies, tns_cols
 
 
def prepare_data(df: pd.DataFrame, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and labels for LGBM model.
    Selects 14 features, groups by trip_id and aggregates with mean.
 
    Args:
        df: Input DataFrame with trip data
        seq_len: Number of latest rows to select per trip (default: 30)
 
    Returns:
        Tuple of (X, y, feature_cols) where X is 2D array, y is 1D array
    """
    df_processed = df.copy()
    # Select only the latest seq_len rows per trip (sorted by 'times')
    df_processed = df_processed.sort_values(["new_trip_id", "times"])
    df_processed = (
        df_processed.groupby("new_trip_id", sort=False)
        .tail(seq_len)
        .reset_index(drop=True)
    )
 
    # 14 selected features for LGBM - Order must match C++ struct exactly!
    selected_cols = [
        "B_P",
        "OTHLDIS",
        "PKB_BDB",
        "SP1",
        "SSA",
        "times",
        "avg_sudden_acceleration_count",
        "avg_harsh_break_count",
        "avg_speed",
        "lc_count",
        "max_speed_continuous_above_130",
        "TTC_filled",
        "lane_change_behavior_flag",
        "unsteady_driving_flag",
    ]
 
    df_processed = df_processed[selected_cols + ['new_trip_id', 'label', 'group_trip_id']]
 
    # Keep track of original trip_id for metadata (to prevent leakage in EI)
    # We take the 'first' because all rows in a new_trip_id come from the same original trip

    metadata_df = df_processed.groupby('new_trip_id')['group_trip_id'].first().reset_index()
    metadata_df.rename(columns={'group_trip_id': 'original_trip_id'}, inplace=True)

    # Aggregate: mean for features, max for label
    agg_dict = {col: 'mean' for col in selected_cols}
    agg_dict.update({'label': 'max'})
 
    df_processed = df_processed.groupby('new_trip_id').agg(agg_dict).reset_index()
   
    # Merge back the original_trip_id
    df_processed = df_processed.merge(metadata_df, on='new_trip_id', how='left')
 
    X = df_processed[selected_cols].values
    y = df_processed['label'].values
    meta = df_processed['original_trip_id'].values
 
    return X, y, selected_cols, meta
 
 
def prepare_data_sequence(df: pd.DataFrame, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features and labels for sequence models (LSTM, iTransformer).
    One-hot encodes categorical column TNS (values 0,1,2,3) to binary columns.
    Returns 3D array of shape (N, seq_len=30, features).
    """
    df_processed = df.copy()
 
    num_cols = [
        "B_P",
        "HV_ACCP",
        "OTHLDIS",
        "PKB_BDB",
        "PMC",
        "SP1",
        "SSA",
        "SSAV",
        "VSC_GX0",
        "VSC_GY0",
        "VSC_YAW0",
        "WSTP",
        "PWC",
        "times",
        "avg_sudden_acceleration_count",
        "avg_harsh_break_count",
        "avg_speed",
        "lc_count",
        "max_speed_continuous_above_130",
        "TTC_filled",
        "lane_change_behavior_flag",
        "unsteady_driving_flag",
    ]
 
    # Encode TNS categorical column
    tns_dummies, tns_cols = _encode_tns_features(df_processed)
 
    # Concatenate one-hot encoded TNS columns with original dataframe
    df_processed = pd.concat([df_processed[['new_trip_id'] + num_cols + ['label']], tns_dummies], axis=1)
 
    y_df = df_processed.groupby('new_trip_id')['label'].max().reset_index()
 
    # Feature columns for sequence model
    feature_cols_list = num_cols + tns_cols
 
    df_encoded = df_processed[['new_trip_id'] + feature_cols_list].copy()
    df_encoded = df_encoded.sort_values(['new_trip_id', 'times'])
 
    X_list = []
    trip_ids = []
    for trip_id, group in df_encoded.groupby('new_trip_id', sort=False):
        group_sorted = group.sort_values('times')
        group_X = group_sorted[feature_cols_list].values
 
        if len(group_X) < seq_len:
            pad_width = ((seq_len - len(group_X), 0), (0, 0))
            group_X = np.pad(group_X, pad_width, mode='constant', constant_values=0)
        else:
            group_X = group_X[-seq_len:]
 
        X_list.append(group_X)
        trip_ids.append(trip_id)
 
    X = np.stack(X_list)
    y_df_dict = y_df.set_index('new_trip_id')['label'].to_dict()
    y = np.array([y_df_dict[trip_id] for trip_id in trip_ids])
 
    feature_cols = feature_cols_list
 
    return X, y, feature_cols
 
 
def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features for 2D input (traditional ML models)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = None
    X_test_scaled = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
 
 
def scale_features_3d(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features for 3D input (N, seq_len, features).
    Scales across the feature dimension while preserving sequence structure.
    """
    N_train, seq_len, n_features = X_train.shape
 
    X_train_2d = X_train.reshape(-1, n_features)
 
    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_train_scaled = X_train_scaled_2d.reshape(N_train, seq_len, n_features)
 
    X_val_scaled = None
    if X_val is not None:
        N_val = X_val.shape[0]
        X_val_2d = X_val.reshape(-1, n_features)
        X_val_scaled_2d = scaler.transform(X_val_2d)
        X_val_scaled = X_val_scaled_2d.reshape(N_val, seq_len, n_features)
 
    X_test_scaled = None
    if X_test is not None:
        N_test = X_test.shape[0]
        X_test_2d = X_test.reshape(-1, n_features)
        X_test_scaled_2d = scaler.transform(X_test_2d)
        X_test_scaled = X_test_scaled_2d.reshape(N_test, seq_len, n_features)
 
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
 
 
def make_weighted_sampler(y: np.ndarray, generator: Any = None) -> Tuple[Any, Dict[int, float]]:
    """
    Create a weighted sampler for handling class imbalance.
   
    Args:
        y: 1-D numpy array of class labels (int), shape (N,)
        generator: Optional torch.Generator for reproducibility
   
    Returns:
        Tuple of (WeightedRandomSampler, class_weights_dict)
    """
    if WeightedRandomSampler is None:
        raise ImportError("PyTorch is required for WeightedRandomSampler")
   
    class_counts = Counter(y.tolist())
    # weight for each class = 1 / count
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[int(label)] for label in y], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator  # Add generator for reproducibility
    )
    return sampler, class_weights
 
 
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_pred_proba: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Evaluate model performance with comprehensive metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
 
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'fpr': fpr,
        'tpr': tpr
    }
 
 
def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
 
 
def plot_roc_curve(results: Dict[str, Any], save_path: str):
    """Plot and save ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for result in results.values():
        # Check if fpr/tpr are available, compute if needed
        if 'fpr' not in result or 'tpr' not in result:
            if 'y_true' in result and 'y_pred_proba' in result:
                fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
                result['fpr'] = fpr
                result['tpr'] = tpr
            else:
                # Skip if we can't compute ROC curve
                continue
       
        fpr, tpr = result['fpr'], result['tpr']
        roc_auc = result['roc_auc']
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.3f})")
 
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
 
 
def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    save_path: str
) -> Tuple[float, float, float]:
    """Plot class distribution for train, val sets."""
    train_counts = np.bincount(y_train.astype(int))
    val_counts = np.bincount(y_val.astype(int))
 
    train_ratio = train_counts[1] / len(y_train) if len(train_counts) > 1 else 0
    val_ratio = val_counts[1] / len(y_val) if len(val_counts) > 1 else 0
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
    datasets = ['Train', 'Validation', 'Test']
    counts_list = [train_counts, val_counts]
    ratios = [train_ratio, val_ratio]
 
    for ax, dataset, counts, ratio in zip(axes, datasets, counts_list, ratios):
        labels = ['Negative (0)', 'Positive (1)']
        if len(counts) == 1:
            counts = np.append(counts, 0)
        ax.bar(labels, counts, color=['skyblue', 'salmon'])
        ax.set_title(f'{dataset} Set\nPositive Ratio: {ratio:.2%}')
        ax.set_ylabel('Count')
        for i, count in enumerate(counts):
            ax.text(i, count + 10, str(count), ha='center')
 
    plt.suptitle('Class Distribution Across Datasets', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
 
    return train_ratio, val_ratio
 
 
def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to file."""
    with open(output_path, 'w') as f:
        for _, result in results.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1-Score: {result['f1_score']:.4f}\n")
            f.write(f"ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write(f"MCC: {result['mcc']:.4f}\n\n")
           
            # Compute confusion matrix if not present
            if 'confusion_matrix' in result:
                cm = result['confusion_matrix']
            elif 'y_true' in result and 'y_pred' in result:
                cm = confusion_matrix(result['y_true'], result['y_pred'])
            else:
                cm = None
           
            if cm is not None:
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n\n")
           
            # Compute classification report if not present
            if 'classification_report' in result:
                f.write("Classification Report:\n")
                f.write(result['classification_report'] + "\n")
            elif 'y_true' in result and 'y_pred' in result:
                class_report = classification_report(result['y_true'], result['y_pred'])
                f.write("Classification Report:\n")
                f.write(class_report + "\n")
 
 
def save_meta_results(
    fold_index: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_results: Dict[str, Any],
    test_results: Dict[str, Any],
    output_path: str
):
    """Save comprehensive meta results including class distribution and all model metrics.
   
    Handles None values for optional parameters (val_ratio, test_ratio, y_val, y_test, val_results, test_results).
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"META RESULTS - FOLD {fold_index}\n")
        f.write("="*80 + "\n\n")
 
        # Class Distribution
        f.write("CLASS DISTRIBUTION\n")
        f.write("="*80 + "\n\n")
 
        # Train Set (always present)
        if y_train is not None:
            train_counts = np.bincount(y_train.astype(int))
            f.write(f"Train Set:\n")
            f.write(f"  Total samples: {len(y_train)}\n")
            f.write(f"  Negative (0): {train_counts[0]} ({(1-train_ratio)*100:.2f}%)\n")
            f.write(f"  Positive (1): {train_counts[1] if len(train_counts) > 1 else 0} ({train_ratio*100:.2f}%)\n\n")
 
        # Validation Set (optional)
        if y_val is not None and val_ratio is not None:
            val_counts = np.bincount(y_val.astype(int))
            f.write(f"Validation Set:\n")
            f.write(f"  Total samples: {len(y_val)}\n")
            f.write(f"  Negative (0): {val_counts[0]} ({(1-val_ratio)*100:.2f}%)\n")
            f.write(f"  Positive (1): {val_counts[1] if len(val_counts) > 1 else 0} ({val_ratio*100:.2f}%)\n\n")
 
        # Test Set (optional)
        if y_test is not None and test_ratio is not None:
            test_counts = np.bincount(y_test.astype(int))
            f.write(f"Test Set:\n")
            f.write(f"  Total samples: {len(y_test)}\n")
            f.write(f"  Negative (0): {test_counts[0]} ({(1-test_ratio)*100:.2f}%)\n")
            f.write(f"  Positive (1): {test_counts[1] if len(test_counts) > 1 else 0} ({test_ratio*100:.2f}%)\n\n")
 
        # Validation Results (optional)
        if val_results is not None and len(val_results) > 0:
            f.write("="*80 + "\n")
            f.write("VALIDATION RESULTS\n")
            f.write("="*80 + "\n\n")
 
            for model_key, result in val_results.items():
                f.write(f"{result['model_name']}:\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n")
                f.write(f"  MCC: {result['mcc']:.4f}\n\n")
 
        # Test Results (optional)
        if test_results is not None and len(test_results) > 0:
            f.write("="*80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n\n")
 
            for model_key, result in test_results.items():
                f.write(f"{result['model_name']}:\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"  ROC-AUC: {result['roc_auc']:.4f}\n")
                f.write(f"  MCC: {result['mcc']:.4f}\n\n")
 
 
def plot_training_history_from_csv(csv_path: str, save_path: str):
    """Plot training history from PyTorch Lightning CSV logger output."""
    metrics_df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
 
    if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
        axes[0].plot(metrics_df['train_loss'].dropna(), label='Train Loss', linewidth=2)
        axes[0].plot(metrics_df['val_loss'].dropna(), label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
 
    if 'train_auc' in metrics_df.columns and 'val_auc' in metrics_df.columns:
        axes[1].plot(metrics_df['train_auc'].dropna(), label='Train AUC', linewidth=2)
        axes[1].plot(metrics_df['val_auc'].dropna(), label='Val AUC', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].set_title('Training AUC')
        axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
 
 
def compute_shap_importance(
    model: Any,
    X_test: np.ndarray,
    X_train: np.ndarray,
    feature_names: List[str],
    model_name: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute mean absolute SHAP values for feature importance.
   
    Args:
        model: Trained ML model
        X_test: Test features array
        X_train: Training features array (for Explainer baseline)
        feature_names: List of feature names
        model_name: Name of the model
   
    Returns:
        Tuple of (mean_abs_shap, importance_df) or (None, None) if computation fails
    """
    try:
        import shap
    except ImportError:
        print("Warning: SHAP is not installed. Install with: pip install shap")
        return None, None
   
    try:
        print(f"  Computing SHAP values for {model_name}...")
       
        # Create SHAP explainer based on model type
        if hasattr(model, 'get_booster'):
            # For XGBoost models
            print(f"  Using XGBoost-specific SHAP configuration...")
            shap_values_plot = None
           
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            except (ValueError, TypeError) as e:
                if "could not convert string to float" in str(e):
                    print(f"  XGBOOST SHAP value computation failed: {e}")
                    print(f"  Using XGBoost native feature importance fallback...")
                    # Fallback to XGBoost's built-in feature importance
                    booster = model.get_booster()
                    importance_dict = booster.get_score(importance_type='gain')
                   
                    # Map feature names (f0, f1, ...) to actual feature names
                    feature_importance = np.zeros(len(feature_names))
                    for key, value in importance_dict.items():
                        if key.startswith('f'):
                            idx = int(key[1:])
                            if idx < len(feature_names):
                                feature_importance[idx] = value
                   
                    # Normalize to make it comparable to SHAP values
                    if feature_importance.sum() > 0:
                        feature_importance = feature_importance / feature_importance.sum()
                   
                    mean_abs_shap = feature_importance
                else:
                    raise
           
            if shap_values_plot is None:
                # If SHAP worked, process the values
                if isinstance(shap_values, list):
                    shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    shap_values_plot = shap_values
               
                # Ensure shap_values_plot is 2D (samples x features)
                if shap_values_plot.ndim == 3:
                    shap_values_plot = shap_values_plot[:, :, -1] if shap_values_plot.shape[2] == 2 else shap_values_plot.reshape(shap_values_plot.shape[0], -1)
                elif shap_values_plot.ndim == 1:
                    shap_values_plot = shap_values_plot.reshape(1, -1)
               
                mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
               
        elif hasattr(model, 'estimators_'):
            # For RandomForest models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # RandomForest returns a list [class_0, class_1], take positive class
            shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
           
            if shap_values_plot.ndim == 3:
                shap_values_plot = shap_values_plot[:, :, -1] if shap_values_plot.shape[2] == 2 else shap_values_plot.reshape(shap_values_plot.shape[0], -1)
            elif shap_values_plot.ndim == 1:
                shap_values_plot = shap_values_plot.reshape(1, -1)
           
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
           
        elif hasattr(model, 'booster_'):
            # For LightGBM models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # LightGBM can return different formats depending on objective
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_plot = shap_values
           
            if shap_values_plot.ndim == 3:
                shap_values_plot = shap_values_plot[:, :, -1] if shap_values_plot.shape[2] == 2 else shap_values_plot.reshape(shap_values_plot.shape[0], -1)
            elif shap_values_plot.ndim == 1:
                shap_values_plot = shap_values_plot.reshape(1, -1)
           
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
           
        elif hasattr(model, 'coef_'):
            # For Logistic Regression and other linear models
            print(f"  Using Linear model SHAP configuration...")
            explainer = shap.Explainer(model, X_train, feature_names=feature_names)
            shap_values = explainer(X_test)
            # For linear models, shap_values is an Explanation object
            if hasattr(shap_values, 'values'):
                if shap_values.values.ndim == 3:
                    # Shape: (samples, features, classes) - take positive class
                    shap_values_plot = shap_values.values[:, :, 1]
                elif shap_values.values.ndim == 2:
                    # Shape: (samples, features)
                    shap_values_plot = shap_values.values
                else:
                    shap_values_plot = shap_values.values
            else:
                shap_values_plot = shap_values
           
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
        else:
            # Fallback for other tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap_values_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
           
            if shap_values_plot.ndim == 3:
                shap_values_plot = shap_values_plot[:, :, -1] if shap_values_plot.shape[2] == 2 else shap_values_plot.reshape(shap_values_plot.shape[0], -1)
            elif shap_values_plot.ndim == 1:
                shap_values_plot = shap_values_plot.reshape(1, -1)
           
            mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
       
        # Ensure mean_abs_shap is 1D
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()
       
        # Verify dimensions match
        if len(mean_abs_shap) != len(feature_names):
            print(f"  Warning: SHAP values dimension ({len(mean_abs_shap)}) doesn't match features ({len(feature_names)})")
            return None, None
       
        # Sort features by importance (descending order)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx]
       
        # Create DataFrame for feature importances
        importance_df = pd.DataFrame({
            'feature': sorted_features,
            'mean_abs_shap_value': sorted_values
        })
       
        print(f"  SHAP values computed successfully for {model_name}")
        return mean_abs_shap, importance_df
       
    except Exception as e:
        print(f"  Warning: Could not compute SHAP values for {model_name}: {e}")
        print(f"  Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None
 
 
def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    model_name: str,
    save_path: str,
    use_shap: bool = False,
    is_shap_data: bool = False
):
    """
    Plot feature importance bar chart.
   
    Args:
        feature_names: List of feature names
        importance_values: Importance scores or SHAP values
        model_name: Name of the model
        save_path: Path to save the plot
        use_shap: If True, label indicates SHAP values
        is_shap_data: If True, importance_values are mean absolute SHAP values
    """
    # Handle dimension mismatches
    if importance_values.ndim > 1:
        if importance_values.ndim == 3:
            # If 3D SHAP values, take mean absolute values and flatten
            importance_values = np.abs(importance_values).mean(axis=0)
            if importance_values.ndim > 1:
                importance_values = importance_values.flatten()
        else:
            importance_values = importance_values.flatten()
   
    # Verify dimensions match
    if len(importance_values) != len(feature_names):
        print(f"Warning: Importance values dimension ({len(importance_values)}) doesn't match features ({len(feature_names)})")
        return None
   
    # Sort features by importance (descending order)
    sorted_idx = np.argsort(importance_values)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = importance_values[sorted_idx]
 
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_values[::-1], color='#1E88E5')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features[::-1])
   
    # Set xlabel based on importance type
    if is_shap_data:
        ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    elif use_shap:
        ax.set_xlabel('SHAP Value')
    else:
        ax.set_xlabel('Importance Score')
   
    # Set title based on importance type
    if is_shap_data:
        ax.set_title(f'SHAP Feature Importance - {model_name} (Test Set)')
    else:
        ax.set_title(f'Feature Importance - {model_name}')
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
   
    return save_path
 
 
def save_feature_importance_csv(
    feature_names: List[str],
    importance_values: np.ndarray,
    save_path: str,
    is_shap_data: bool = False
):
    """
    Save feature importance to CSV file.
   
    Args:
        feature_names: List of feature names
        importance_values: Importance scores or SHAP values
        save_path: Path to save CSV file
        is_shap_data: If True, column name is 'mean_abs_shap_value'
    """
    # Handle dimension mismatches
    if importance_values.ndim > 1:
        if importance_values.ndim == 3:
            # If 3D SHAP values, take mean absolute values and flatten
            importance_values = np.abs(importance_values).mean(axis=0)
            if importance_values.ndim > 1:
                importance_values = importance_values.flatten()
        else:
            importance_values = importance_values.flatten()
   
    # Verify dimensions match
    if len(importance_values) != len(feature_names):
        print(f"Warning: Importance values dimension ({len(importance_values)}) doesn't match features ({len(feature_names)})")
        return None
   
    # Determine column name based on importance type
    column_name = 'mean_abs_shap_value' if is_shap_data else 'importance'
   
    importance_df = pd.DataFrame({
        'feature': feature_names,
        column_name: importance_values
    })
    importance_df = importance_df.sort_values(column_name, ascending=False).reset_index(drop=True)
    importance_df.to_csv(save_path, index=False)
   
    return save_path
 
def find_best_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """
    Find the best threshold that maximizes the F1-score.
 
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for the positive class
 
    Returns:
        best_threshold: The threshold that maximizes the F1-score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold
 
def plot_class_distribution_final(y_train_val: np.ndarray, y_test: np.ndarray, save_path: str) -> Tuple[float, float]:
    """
    Plot class distribution for final model (train+val and test).
   
    Args:
        y_train_val: Combined train and validation labels
        y_test: Test labels
        save_path: Path to save the plot
       
    Returns:
        Tuple of (train_val_positive_ratio, test_positive_ratio)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
   
    # Train+Val distribution
    train_val_pos = np.sum(y_train_val == 1)
    train_val_neg = np.sum(y_train_val == 0)
    train_val_ratio = train_val_pos / len(y_train_val)
   
    axes[0].bar(['Negative', 'Positive'], [train_val_neg, train_val_pos], color=['blue', 'orange'])
    axes[0].set_title(f'Train+Val Distribution (n={len(y_train_val)})')
    axes[0].set_ylabel('Count')
    axes[0].text(0, train_val_neg, f'{train_val_neg}\n({(1-train_val_ratio)*100:.1f}%)',
                ha='center', va='bottom')
    axes[0].text(1, train_val_pos, f'{train_val_pos}\n({train_val_ratio*100:.1f}%)',
                ha='center', va='bottom')
   
    # Test distribution
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)
    test_ratio = test_pos / len(y_test)
   
    axes[1].bar(['Negative', 'Positive'], [test_neg, test_pos], color=['blue', 'orange'])
    axes[1].set_title(f'Test Distribution (n={len(y_test)})')
    axes[1].set_ylabel('Count')
    axes[1].text(0, test_neg, f'{test_neg}\n({(1-test_ratio)*100:.1f}%)',
                ha='center', va='bottom')
    axes[1].text(1, test_pos, f'{test_pos}\n({test_ratio*100:.1f}%)',
                ha='center', va='bottom')
   
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
   
    return train_val_ratio, test_ratio
 
 
def log_and_save_oof_results(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    oof_threshold: float,
    model_name: str,
    output_dir: Path,
    logger: Any,
    oof_df: pd.DataFrame = None,
    oof_merged_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Log and save OOF (Out-Of-Fold) evaluation results.
   
    This is a reusable function for evaluating OOF predictions from K-fold cross-validation,
    logging the results in a formatted way, and saving them to disk.
   
    Args:
        y_true: True labels from concatenated K-fold validation sets
        y_pred_proba: Predicted probabilities from concatenated K-fold validation sets
        oof_threshold: Optimal threshold computed on OOF predictions
        model_name: Name of the model (e.g., "LGBM", "XGB", "RF")
        output_dir: Base output directory where "oof" subdirectory will be created
        logger: Logger instance for logging results
        oof_df: Optional DataFrame with columns ['new_trip_id', 'y_pred_proba'] for saving predictions
        oof_merged_df: Optional DataFrame with predictions merged with true labels for saving
       
    Returns:
        Dictionary containing evaluation metrics (same format as evaluate_model)
    """
    # Compute predictions using threshold
    y_pred_labels = (y_pred_proba >= oof_threshold).astype(int)
   
    # Evaluate model
    oof_result = evaluate_model(
        y_true, y_pred_labels, y_pred_proba, f"{model_name}_OOF"
    )
   
    # Log results with formatted output
    logger.info("")
    logger.info("=" * 80)
    logger.info("OOF EVALUATION RESULTS (K-Fold Cross-Validation)")
    logger.info("=" * 80)
    logger.info("Model: %s", model_name)
    logger.info("OOF Validation Threshold: %.4f", oof_threshold)
    logger.info("Precision:  %.4f", oof_result['precision'])
    logger.info("Recall:     %.4f", oof_result['recall'])
    logger.info("F1 Score:   %.4f", oof_result['f1_score'])
    logger.info("ROC AUC:    %.4f", oof_result['roc_auc'])
    logger.info("MCC:        %.4f", oof_result['mcc'])
    logger.info("Accuracy:   %.4f", oof_result['accuracy'])
    logger.info("=" * 80)
   
    # Save results to disk
    if output_dir is not None:
        oof_output_dir = Path(output_dir) / "oof"
        oof_output_dir.mkdir(parents=True, exist_ok=True)
       
        # Save evaluation metrics
        oof_results_path = oof_output_dir / "oof_results.txt"
        oof_result_save = {
            f"oof_{model_name.lower()}": oof_result
        }
        save_results(oof_result_save, str(oof_results_path))
        logger.info("OOF evaluation results saved to %s", oof_results_path)
       
        # Save OOF predictions if provided
        if oof_df is not None:
            oof_pred_path = oof_output_dir / "oof_val_predictions.csv"
            oof_df.to_csv(oof_pred_path, index=False)
            logger.info("OOF predictions saved to %s", oof_pred_path)
       
        # Save merged predictions (with true labels) if provided
        if oof_merged_df is not None:
            oof_merged_path = oof_output_dir / "oof_val_predictions_merged.csv"
            oof_merged_df.to_csv(oof_merged_path, index=False)
            logger.info("OOF predictions (merged with labels) saved to %s", oof_merged_path)
   
    return oof_result