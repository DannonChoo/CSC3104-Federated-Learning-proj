
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

def build_preprocess(df: pd.DataFrame):
    # Exclude leakage/labels from features
    feature_cols = [c for c in df.columns if c not in EXCLUDE_AS_FEATURES]

    # Drop feature columns that are >60% null OR constant (nunique <= 1)
    to_drop = []
    for c in feature_cols:
        null_pct = df[c].isna().mean()
        if null_pct > 0.60 or df[c].nunique(dropna=True) <= 1:
            to_drop.append(c)
    if to_drop:
        feature_cols = [c for c in feature_cols if c not in to_drop]

    # Split into features/targets
    X = df[feature_cols].copy()
    Y = df[TARGET_COLS].astype(int).values  # shape (N, n_tasks)

    # Separate types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # Optional: winsorize numeric to 1st–99th percentile to reduce extreme outliers
    if num_cols:
        q_low = X[num_cols].quantile(0.01)
        q_hi  = X[num_cols].quantile(0.99)
        X[num_cols] = X[num_cols].clip(lower=q_low, upper=q_hi, axis=1)

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

    Xt = pre.fit_transform(X)

    info = {
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dropped_cols_high_null_or_constant": to_drop,
        "transformed_shape": [int(Xt.shape[0]), int(Xt.shape[1])],
    }
    print(f"[data_prep] Features kept: {len(feature_cols)} | dropped: {len(to_drop)} | transformed to {Xt.shape}")
    return pre, Xt, Y, info

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

    # Load & EDA
    df = load_dataframe()
    eda = eda_summary(df, TARGET_COLS)

    # Save EDA
    with open(DATA_DIR / "eda_report.json", "w") as f:
        json.dump(eda, f, indent=2)
    print("[data_prep] EDA summary saved to data/eda_report.json")
    # Brief print
    chd_col = TARGET_COLS[0]  # assume order aligns with HEADS (chd first)
    pos_rate = float(df[chd_col].mean())
    print(f"[data_prep] Rows={len(df)} | CHD Positives={df[chd_col].sum()} | CHD PosRate={pos_rate:.4f}")

    # Build preprocessing & transform
    pre, Xt_full, Y_full, prep_info = build_preprocess(df)

    # Create a proper holdout (20%) before federated sharding for eval
    idx_all = np.arange(len(Y_full))
    # Stratify by CHD (first target) for split stability
    idx_train, idx_hold = train_test_split(
        idx_all, test_size=0.20, stratify=Y_full[:, 0], random_state=42
    )
    Xt_train, Y_train = Xt_full[idx_train], Y_full[idx_train]
    Xt_hold,  Y_hold  = Xt_full[idx_hold],  Y_full[idx_hold]

    # Save preprocessing pipeline and specs
    dump(pre, DATA_DIR / "preprocess.joblib")
    spec = {
        "feature_cols": prep_info["feature_cols"],
        "num_cols": prep_info["num_cols"],
        "cat_cols": prep_info["cat_cols"],
        "target_cols": TARGET_COLS,
        "heads": HEADS,
        "dropped_cols_high_null_or_constant": prep_info["dropped_cols_high_null_or_constant"],
        "transformed_dim": prep_info["transformed_shape"][1],
    }
    with open(DATA_DIR / "feature_spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    print("[data_prep] Saved preprocess.joblib and feature_spec.json")

    # Save holdout set for server-side evaluation
    np.savez(DATA_DIR / "holdout.npz", X=Xt_hold, y=Y_hold)
    print(f"[data_prep] Saved holdout.npz: X={Xt_hold.shape}, y_pos_each={[int(Y_hold[:,i].sum()) for i in range(Y_hold.shape[1])]}, n={len(Y_hold)}")

    # Build federated shards from TRAIN ONLY (stratify by CHD index 0)
    shards = make_non_iid_splits(Xt_train, Y_train, n_sites=NUM_CLIENTS, stratify_idx=0)
    for site_id, idx in enumerate(shards):
        X_site = Xt_train[idx]
        Y_site = Y_train[idx]
        np.savez(DATA_DIR / f"site_{site_id}.npz", X=X_site, y=Y_site)
        pos_each = [int(Y_site[:,i].sum()) for i in range(Y_site.shape[1])]
        print(f"[data_prep] Saved site_{site_id}.npz: X={X_site.shape}, y_pos_each={pos_each}, n={len(Y_site)}")

    # Save a brief prep report
    prep_report = {
        "num_clients": NUM_CLIENTS,
        "train_rows": int(len(Y_train)),
        "holdout_rows": int(len(Y_hold)),
        "train_pos_rate_CHD": float(Y_train[:,0].mean()),
        "holdout_pos_rate_CHD": float(Y_hold[:,0].mean()),
        "transformed_dim": int(Xt_full.shape[1]),
    }
    with open(DATA_DIR / "prep_report.json", "w") as f:
        json.dump(prep_report, f, indent=2)
    print("[data_prep] prep_report.json written. Done.")

if __name__ == "__main__":
    main()
