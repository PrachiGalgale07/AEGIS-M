"""
cmaps_utils.py
--------------
Utilities to load, clean, and window the NASA C-MAPSS turbofan degradation
dataset (FD001, FD002, FD003, FD004) for Remaining-Useful-Life (RUL)
regression + binary "at risk" classification.

Expected folder layout (place your real data here):

    data/CMaps/
        train_FD001.txt   test_FD001.txt   RUL_FD001.txt
        train_FD002.txt   test_FD002.txt   RUL_FD002.txt
        train_FD003.txt   test_FD003.txt   RUL_FD003.txt
        train_FD004.txt   test_FD004.txt   RUL_FD004.txt

Each train/test file is whitespace-separated with NO header and columns:
    unit_number, time_in_cycles, op_setting_1, op_setting_2, op_setting_3,
    sensor_1 ... sensor_21
RUL_FD00X.txt has one integer per line = true RUL at the LAST cycle of the
corresponding unit in test_FD00X.txt (in unit order).
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FD_IDS = ["FD001", "FD002", "FD003", "FD004"]

BASE_COLS = ["unit", "cycle", "op1", "op2", "op3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
COLUMN_NAMES = BASE_COLS + SENSOR_COLS

RUL_CAP = 125          # standard piecewise-linear RUL cap used in C-MAPSS literature
RISK_THRESHOLD = 30    # cycles remaining -> considered "at risk" (binary label = 1)
N_OP_CLUSTERS = 6      # FD002/FD004 have 6 real operating regimes; FD001/FD003 collapse to ~1


def _read_raw(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES
    return df


def _find_file(root, filename):
    """Look for `filename` directly inside root, then fall back to a
    recursive search anywhere under root (handles extra nesting like
    CMaps/CMaps/train_FD001.txt)."""
    direct = os.path.join(root, filename)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
    return matches[0] if matches else None


def find_cmaps_root(candidates):
    """Given a list of candidate folder paths, return the first one that
    actually contains train_FD00X.txt files anywhere inside it (at any
    depth). Returns None if nothing is found."""
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "**", "train_FD*.txt"), recursive=True):
            return c
    return None


def load_all_train(data_dir, fd_ids=FD_IDS):
    """Load and concatenate train_FD00X.txt for every requested subset.
    Searches recursively under data_dir so it works whether files sit
    directly in the folder or one level deeper."""
    frames = []
    for fd in fd_ids:
        path = _find_file(data_dir, f"train_{fd}.txt")
        if path is None:
            print(f"[warn] could not find train_{fd}.txt anywhere under {data_dir}, skipping")
            continue
        df = _read_raw(path)
        df["dataset_id"] = fd
        # make unit ids globally unique across subsets
        df["global_unit"] = fd + "_" + df["unit"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No train_FD00X.txt files found anywhere under {data_dir}. "
            "Download the C-MAPSS dataset and place all four subsets there."
        )
    return pd.concat(frames, ignore_index=True)


def load_all_test(data_dir, fd_ids=FD_IDS):
    """Load test_FD00X.txt + matching RUL_FD00X.txt for every requested subset."""
    frames = []
    for fd in fd_ids:
        tpath = _find_file(data_dir, f"test_{fd}.txt")
        rpath = _find_file(data_dir, f"RUL_{fd}.txt")
        if tpath is None or rpath is None:
            print(f"[warn] could not find test_{fd}.txt / RUL_{fd}.txt under {data_dir}, skipping")
            continue
        df = _read_raw(tpath)
        df["dataset_id"] = fd
        df["global_unit"] = fd + "_" + df["unit"].astype(str)

        true_rul = pd.read_csv(rpath, header=None)[0].values
        units = sorted(df["unit"].unique())
        if len(units) != len(true_rul):
            print(f"[warn] {fd}: unit count {len(units)} != RUL rows {len(true_rul)}")
        rul_map = {u: r for u, r in zip(units, true_rul)}
        df["final_rul"] = df["unit"].map(rul_map)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No test/RUL files found under {data_dir}.")
    return pd.concat(frames, ignore_index=True)


def add_train_rul(df, cap=RUL_CAP):
    """Add piecewise-linear RUL label for training data (RUL known exactly per unit)."""
    max_cycle = df.groupby("global_unit")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    df = df.copy()
    df["RUL"] = np.clip(rul, 0, cap)
    return df


def add_test_rul(df, cap=RUL_CAP):
    """Add RUL label for test data: final_rul is RUL at the LAST observed cycle,
    so RUL at any earlier cycle = final_rul + (last_cycle - cycle)."""
    df = df.copy()
    last_cycle = df.groupby("global_unit")["cycle"].transform("max")
    df["RUL"] = np.clip(df["final_rul"] + (last_cycle - df["cycle"]), 0, cap)
    return df


def add_binary_label(df, threshold=RISK_THRESHOLD):
    df = df.copy()
    df["risk"] = (df["RUL"] < threshold).astype(int)
    return df


def fit_op_condition_clusters(train_df, n_clusters=N_OP_CLUSTERS, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(train_df[OP_COLS].values)
    return kmeans


def fit_sensor_scalers(train_df, kmeans):
    """Fit one StandardScaler per operating-condition cluster (handles FD002/FD004
    multi-regime data correctly, and degenerates to a single scaler for FD001/FD003)."""
    clusters = kmeans.predict(train_df[OP_COLS].values)
    scalers = {}
    tmp = train_df.copy()
    tmp["op_cond"] = clusters
    for c in sorted(tmp["op_cond"].unique()):
        sub = tmp[tmp["op_cond"] == c]
        scaler = StandardScaler()
        scaler.fit(sub[SENSOR_COLS].values)
        scalers[int(c)] = scaler
    op_scaler = StandardScaler()
    op_scaler.fit(train_df[OP_COLS].values)
    return scalers, op_scaler


def transform_features(df, kmeans, sensor_scalers, op_scaler, n_clusters=N_OP_CLUSTERS):
    """Produce the final feature matrix: normalized op settings + normalized sensors
    (per operating-condition cluster) + one-hot operating-condition cluster."""
    df = df.copy()
    df["op_cond"] = kmeans.predict(df[OP_COLS].values)

    normed_sensors = np.zeros((len(df), len(SENSOR_COLS)))
    for c, scaler in sensor_scalers.items():
        mask = (df["op_cond"] == c).values
        if mask.any():
            normed_sensors[mask] = scaler.transform(df.loc[mask, SENSOR_COLS].values)

    normed_ops = op_scaler.transform(df[OP_COLS].values)

    onehot = np.zeros((len(df), n_clusters))
    onehot[np.arange(len(df)), df["op_cond"].values.astype(int)] = 1.0

    features = np.concatenate([normed_ops, normed_sensors, onehot], axis=1)
    feature_names = OP_COLS + SENSOR_COLS + [f"op_cond_{i}" for i in range(n_clusters)]
    return features, feature_names


def build_sequences(df, features, window_size, group_col="global_unit",
                     rul_col="RUL", risk_col="risk", last_only=False):
    """Slide a fixed-size window over each unit's trajectory (sorted by cycle).
    Units shorter than window_size are front-padded by repeating the first row,
    so short trajectories (common in FD002/FD004) are not thrown away.

    If last_only=True, only the final window per unit is returned (used to
    score against the official test RUL_FD00X.txt values).
    """
    df = df.reset_index(drop=True).copy()
    df["_feat_idx"] = np.arange(len(df))

    X, y_rul, y_risk, groups = [], [], [], []
    for gid, g in df.groupby(group_col, sort=False):
        g = g.sort_values("cycle")
        idx = g["_feat_idx"].values
        n = len(idx)
        feats = features[idx]
        ruls = g[rul_col].values
        risks = g[risk_col].values if risk_col in g.columns else np.zeros(n)

        if n < window_size:
            pad = window_size - n
            feats = np.vstack([np.repeat(feats[:1], pad, axis=0), feats])
            ruls = np.concatenate([np.repeat(ruls[:1], pad), ruls])
            risks = np.concatenate([np.repeat(risks[:1], pad), risks])
            n = window_size

        if last_only:
            starts = [n - window_size]
        else:
            starts = range(0, n - window_size + 1)

        for s in starts:
            X.append(feats[s:s + window_size])
            y_rul.append(ruls[s + window_size - 1])
            y_risk.append(risks[s + window_size - 1])
            groups.append(gid)

    return (np.array(X, dtype=np.float32),
            np.array(y_rul, dtype=np.float32),
            np.array(y_risk, dtype=np.float32),
            np.array(groups))


def save_preprocessing(path, kmeans, sensor_scalers, op_scaler, feature_names,
                        window_size, risk_threshold=RISK_THRESHOLD, rul_cap=RUL_CAP):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "kmeans.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(path, "sensor_scalers.pkl"), "wb") as f:
        pickle.dump(sensor_scalers, f)
    with open(os.path.join(path, "op_scaler.pkl"), "wb") as f:
        pickle.dump(op_scaler, f)
    config = {
        "feature_names": feature_names,
        "window_size": window_size,
        "risk_threshold": risk_threshold,
        "rul_cap": rul_cap,
        "n_op_clusters": len(sensor_scalers),
        "sensor_cols": SENSOR_COLS,
        "op_cols": OP_COLS,
    }
    with open(os.path.join(path, "rul_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_preprocessing(path):
    with open(os.path.join(path, "kmeans.pkl"), "rb") as f:
        kmeans = pickle.load(f)
    with open(os.path.join(path, "sensor_scalers.pkl"), "rb") as f:
        sensor_scalers = pickle.load(f)
    with open(os.path.join(path, "op_scaler.pkl"), "rb") as f:
        op_scaler = pickle.load(f)
    with open(os.path.join(path, "rul_config.json")) as f:
        config = json.load(f)
    return kmeans, sensor_scalers, op_scaler, config
