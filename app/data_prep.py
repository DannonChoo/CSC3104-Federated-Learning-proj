
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import multi-task settings (TARGET_COLS, HEADS) and common config
from config import (
    CSV_PATH, DATA_DIR, DROP_COLS, EXCLUDE_AS_FEATURES, NUM_CLIENTS,
    # Multi-task:
    TARGET_COLS, HEADS,
)

# ---------- Helper: basic EDA summaries (multi-target) ----------

def eda_summary(df: pd.DataFrame, targets) -> dict:
    n_rows = len(df)
    nulls = df.isna().sum().to_dict()
    null_pct = (df.isna().sum() / max(n_rows, 1) * 100).round(2).to_dict()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    # Per-target stats
    target_stats = {}
    for t in targets:
        vc = df[t].value_counts(dropna=False).to_dict()
        pos_rate = float(df[t].mean()) if pd.api.types.is_numeric_dtype(df[t]) else None
        imbalanced = None
        if pos_rate is not None:
            tail = min(pos_rate, 1.0 - pos_rate)
            imbalanced = tail < 0.10  # flag if either class <10%

        target_stats[t] = {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "positive_rate": pos_rate,
            "minority_rate": (None if pos_rate is None else float(tail)),
            "imbalanced_flag": imbalanced,
            "imbalance_side": (
                None if pos_rate is None else ("minority-positive" if pos_rate < 0.5 else "minority-negative")
            ),
        }

    return {
        "rows": int(n_rows),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "null_counts": nulls,
        "null_pct": null_pct,
        "nunique": nunique,
        "targets": targets,
        "per_target": target_stats,
    }

# ---------- Load & clean ----------

def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Drop explicitly requested columns
    drops = [c for c in DROP_COLS if c in df.columns]
    if drops:
        df = df.drop(columns=drops)

    # Ensure all TARGET_COLS exist
    missing = [t for t in TARGET_COLS if t not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in CSV: {missing}")

    # Keep rows with ALL targets present and cast to int
    df = df.dropna(subset=TARGET_COLS).copy()
    for t in TARGET_COLS:
        df[t] = df[t].astype(int)

    return df

# ---------- Feature selection & preprocessing ----------

def build_preprocess_for_train(df_train: pd.DataFrame, feature_cols):
    Xtr = df_train[feature_cols].copy()

    # Separate types
    num_cols = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Light winsorization ON TRAIN ONLY (optional)
    if num_cols:
        q_low = Xtr[num_cols].quantile(0.01)
        q_hi  = Xtr[num_cols].quantile(0.99)
        Xtr[num_cols] = Xtr[num_cols].clip(lower=q_low, upper=q_hi, axis=1)

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])

    # Fit on TRAIN ONLY
    pre.fit(Xtr)

    return pre, num_cols, cat_cols

# ---------- Federated shards (non-IID, stratified by a primary label) + holdout ----------

def make_non_iid_splits(Xt, Y, n_sites, stratify_idx=0):
    # Create n_sites shards using stratification on one primary label (default: CHD). CHD is typically the primary outcome.
    y_primary = Y[:, stratify_idx]
    skf = StratifiedKFold(n_splits=n_sites, shuffle=True, random_state=42)
    shards = []
    for _, idx in skf.split(Xt, y_primary):
        shards.append(idx)
    return shards

def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # 1) Load raw, drop columns, ensure targets are present/typed
    df = load_dataframe()

    # EDA
    eda = eda_summary(df, TARGET_COLS)
    (DATA_DIR / "eda_report.json").write_text(json.dumps(eda, indent=2))
    print("[data_prep] EDA summary saved to data/eda_report.json")

    # Feature selection (exclude leakage)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_AS_FEATURES]

    # Drop bad features (>60% null or constant)
    to_drop = []
    for c in feature_cols:
        null_pct = df[c].isna().mean()
        if null_pct > 0.60 or df[c].nunique(dropna=True) <= 1:
            to_drop.append(c)
    if to_drop:
        feature_cols = [c for c in feature_cols if c not in to_drop]

    # Targets as a 2D array
    Y_full = df[TARGET_COLS].astype(int).values

    # Split indices FIRST (stratify by CHD = Y_full[:,0])
    import numpy as np
    from sklearn.model_selection import train_test_split, StratifiedKFold

    idx_all = np.arange(len(Y_full))
    idx_train, idx_hold = train_test_split(
        idx_all, test_size=0.20, stratify=Y_full[:, 0], random_state=42
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_hold  = df.iloc[idx_hold].reset_index(drop=True)

    # Fit preprocessor on TRAIN
    pre, num_cols, cat_cols = build_preprocess_for_train(df_train, feature_cols)

    # Transform train/holdout separately
    def apply_transform(df_subset):
        Xs = df_subset[feature_cols].copy()
        # If you kept winsorization inside build_preprocess_for_train, you can
        # optionally duplicate the clip here OR skip it entirely (apply only on train).
        return pre.transform(Xs)

    Xt_train = apply_transform(df_train)
    Xt_hold  = apply_transform(df_hold)
    Y_train  = df_train[TARGET_COLS].astype(int).values
    Y_hold   = df_hold[TARGET_COLS].astype(int).values

    # Persist preprocessor and spec
    from joblib import dump
    dump(pre, DATA_DIR / "preprocess.joblib")
    spec = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_cols": TARGET_COLS,
        "heads": HEADS,
        "dropped_cols_high_null_or_constant": to_drop,
        "transformed_dim": int(Xt_train.shape[1]),
    }
    (DATA_DIR / "feature_spec.json").write_text(json.dumps(spec, indent=2))
    print("[data_prep] Saved preprocess.joblib and feature_spec.json")

    # Save a clean holdout (server evaluation is now trustworthy)
    np.savez(DATA_DIR / "holdout.npz", X=Xt_hold, y=Y_hold)

    # Build federated shards from TRAIN ONLY
    skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=42)
    y_primary = Y_train[:, 0]
    for site_id, (_, idx) in enumerate(skf.split(Xt_train, y_primary)):
        X_site = Xt_train[idx]
        Y_site = Y_train[idx]
        np.savez(DATA_DIR / f"site_{site_id}.npz", X=X_site, y=Y_site)
        print(f"[data_prep] Saved site_{site_id}.npz: X={X_site.shape}, n={len(Y_site)}")

    print("[data_prep] Done.")

if __name__ == "__main__":
    main()
