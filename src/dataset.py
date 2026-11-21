import os
import json
import numpy as np
import pandas as pd

def load_dataset(data_dir="data", config_dir="configs", raw_csv_path=None):
    """
    Loads raw CICMalDroid 2020 CSV and reconstructs the 474-D feature space
    using the split-first dev/holdout index partition.
    """
    if raw_csv_path is None:
        # Search common local paths
        candidates = [
            "data/feature_vectors_syscallsbinders_frequency_5_Cat.csv",
            "feature_vectors_syscallsbinders_frequency_5_Cat.csv",
            "../data/feature_vectors_syscallsbinders_frequency_5_Cat.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                raw_csv_path = c
                break
    
    if raw_csv_path is None or not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            "Raw CICMalDroid CSV not found! Place 'feature_vectors_syscallsbinders_frequency_5_Cat.csv' "
            "inside the data/ folder or pass raw_csv_path explicitly."
        )

    df_raw = pd.read_csv(raw_csv_path)
    label_col = "Class" if "Class" in df_raw.columns else df_raw.columns[-1]
    y_raw = pd.to_numeric(df_raw[label_col], errors="raise").astype(int).to_numpy()
    if y_raw.min() == 1:
        y_raw = y_raw - 1  # 0-indexed: 0..4

    X_raw = df_raw.drop(columns=[label_col])

    # Load feature group keywords
    fg_path = os.path.join(config_dir, "feature_groups.json")
    with open(fg_path, "r", encoding="utf-8") as fh:
        fg_json = json.load(fh)

    # 474-D Domain Feature Engineering (log1p + 4 domain composites)
    X_eng = np.log1p(X_raw)
    X_eng["COMPOSITE_PRIVACY_RISK"]  = np.log1p(X_raw[fg_json["privacy_cols"]].sum(axis=1))
    X_eng["COMPOSITE_MEMORY_TAMPER"] = np.log1p(X_raw[fg_json["memory_cols"]].sum(axis=1))
    X_eng["COMPOSITE_IPC_BINDER"]    = np.log1p(X_raw[fg_json["ipc_cols"]].sum(axis=1))
    X_eng["COMPOSITE_IO_ACTIVITY"]   = np.log1p(X_raw[fg_json["io_cols"]].sum(axis=1))

    # Load split-first partition indices
    dev_idx  = np.load(os.path.join(data_dir, "dev_indices.npy"))
    hold_idx = np.load(os.path.join(data_dir, "holdout_indices.npy"))

    # Load Top-60 feature list
    top60_path = os.path.join(config_dir, "top60_features.json")
    with open(top60_path, "r", encoding="utf-8") as fh:
        t60_meta = json.load(fh)
    top60_cols = t60_meta["top60"] if "top60" in t60_meta else t60_meta["top60_features_ordered"]

    X_dev_60   = X_eng.iloc[dev_idx][top60_cols].to_numpy(dtype=np.float32)
    y_dev      = y_raw[dev_idx]
    X_hold_60  = X_eng.iloc[hold_idx][top60_cols].to_numpy(dtype=np.float32)
    y_hold     = y_raw[hold_idx]

    return X_dev_60, y_dev, X_hold_60, y_hold, top60_cols, X_raw, dev_idx, hold_idx