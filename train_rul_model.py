"""
train_rul_model.py
-------------------
Trains ONE combined deep-learning model on ALL FOUR C-MAPSS subsets
(FD001, FD002, FD003, FD004) that jointly predicts:

  1. RUL (Remaining Useful Life, regression, in cycles)
  2. risk (binary classification: 1 = failure likely within RISK_THRESHOLD cycles)

Run:
    python train_rul_model.py

Expects data at:  data/CMaps/{train,test,RUL}_FD00{1,2,3,4}.txt
Saves outputs to: models/rul/
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers

import cmaps_utils as cu

# Auto-detect where the C-MAPSS files actually live: tries these in order,
# and also searches recursively inside each, so it works whether your
# folder is "data/CMaps", "CMaps", or the files sit one level deeper.
CANDIDATE_DATA_DIRS = [
    os.path.join("data", "CMaps"),
    "CMaps",
    "data",
    ".",
]
MODEL_DIR = os.path.join("models", "rul")
WINDOW_SIZE = 30
BATCH_SIZE = 128
EPOCHS = 60
SEED = 42

FD_IDS = cu.FD_IDS  # ["FD001", "FD002", "FD003", "FD004"] -- ALL subsets used


def build_model(n_timesteps, n_features):
    inp = keras.Input(shape=(n_timesteps, n_features), name="sensor_window")
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    shared = layers.Dense(32, activation="relu")(x)

    rul_out = layers.Dense(1, activation="relu", name="rul")(shared)
    risk_out = layers.Dense(1, activation="sigmoid", name="risk")(shared)

    model = keras.Model(inputs=inp, outputs=[rul_out, risk_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"rul": "mse", "risk": "binary_crossentropy"},
        loss_weights={"rul": 1.0, "risk": 5.0},
        metrics={"rul": ["mae"], "risk": ["accuracy"]},
    )
    return model


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_dir = cu.find_cmaps_root(CANDIDATE_DATA_DIRS)
    if data_dir is None:
        raise FileNotFoundError(
            "Could not locate the C-MAPSS train_FD00X.txt files.\n"
            f"Looked in: {CANDIDATE_DATA_DIRS} (recursively).\n"
            "Fix: either move your CMaps folder so it's at 'CMaps/' or "
            "'data/CMaps/' relative to this script, or edit "
            "CANDIDATE_DATA_DIRS at the top of train_rul_model.py to add "
            "the exact path where your train_FD001.txt etc. live."
        )
    print(f"Using CMaps data folder: {os.path.abspath(data_dir)}")

    print(f"Loading train data for subsets: {FD_IDS}")
    train_df = cu.load_all_train(data_dir, FD_IDS)
    train_df = cu.add_train_rul(train_df)
    train_df = cu.add_binary_label(train_df)
    print(f"  -> {train_df['global_unit'].nunique()} engine units, {len(train_df)} rows total")

    print("Fitting operating-condition clusters + per-cluster sensor scalers ...")
    kmeans = cu.fit_op_condition_clusters(train_df)
    sensor_scalers, op_scaler = cu.fit_sensor_scalers(train_df, kmeans)

    features, feature_names = cu.transform_features(train_df, kmeans, sensor_scalers, op_scaler)
    print(f"  -> {len(feature_names)} input features per timestep")

    print(f"Building sliding windows (window_size={WINDOW_SIZE}) ...")
    X, y_rul, y_risk, groups = cu.build_sequences(train_df, features, WINDOW_SIZE)
    print(f"  -> {X.shape[0]} training sequences of shape {X.shape[1:]}")

    # group-aware split so the same engine unit never appears in both train & val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X, groups=groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_rul_train, y_rul_val = y_rul[train_idx], y_rul[val_idx]
    y_risk_train, y_risk_val = y_risk[train_idx], y_risk[val_idx]

    print(f"Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")
    print(f"Risk-positive rate -> train: {y_risk_train.mean():.3f}, val: {y_risk_val.mean():.3f}")

    model = build_model(WINDOW_SIZE, X.shape[-1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    model.fit(
        X_train,
        {"rul": y_rul_train, "risk": y_risk_train},
        validation_data=(X_val, {"rul": y_rul_val, "risk": y_risk_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Evaluate on the official held-out test sets (all 4 subsets) ----
    try:
        print("\nEvaluating on official test_FD00X + RUL_FD00X files ...")
        test_df = cu.load_all_test(data_dir, FD_IDS)
        test_df = cu.add_test_rul(test_df)
        test_df = cu.add_binary_label(test_df)
        test_features, _ = cu.transform_features(test_df, kmeans, sensor_scalers, op_scaler)
        Xt, yt_rul, yt_risk, _ = cu.build_sequences(
            test_df, test_features, WINDOW_SIZE, last_only=True
        )
        results = model.evaluate(Xt, {"rul": yt_rul, "risk": yt_risk}, verbose=0)
        print("Test set results:", dict(zip(model.metrics_names, results)))
    except FileNotFoundError as e:
        print(f"[warn] skipping official test evaluation: {e}")

    # ---- Save model + preprocessing artifacts ----
    model.save(os.path.join(MODEL_DIR, "rul_model.keras"))
    cu.save_preprocessing(
        MODEL_DIR, kmeans, sensor_scalers, op_scaler, feature_names,
        WINDOW_SIZE, cu.RISK_THRESHOLD, cu.RUL_CAP,
    )
    print(f"\nSaved model + preprocessing artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
