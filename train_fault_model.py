"""
train_fault_model.py
---------------------
For EVERY category found under data/MVTec_AD/ (bottle, cable, capsule,
carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush,
transistor, wood, zipper, ... -- whatever you have), trains two models:

  1. A convolutional autoencoder, trained ONLY on normal ("good") images.
     At inference, pixels that reconstruct poorly indicate a defect ->
     used to localize the damaged area (segmentation/heatmap) and to flag
     "defective vs OK" via a reconstruction-error threshold.

  2. A small CNN classifier (transfer learning on MobileNetV2), trained on
     the labeled test/<defect_type> folders, to name WHICH defect type is
     present (e.g. "scratch", "crack", "broken_large", "bent_wire" ...).

Run:
    python train_fault_model.py

Expects data at: data/MVTec_AD/<category>/{train,test,ground_truth}/...
Saves outputs to: models/fault/<category>/
"""

import json
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import mvtec_utils as mu

# Auto-detect where the MVTec AD category folders actually live: tries
# these in order (and searches inside each), so it works whether your
# folder is "data/MVTec_AD", "MVTec_AD", or nested one level deeper.
CANDIDATE_DATA_DIRS = [
    os.path.join("data", "MVTec_AD"),
    "MVTec_AD",
    "data",
    ".",
]
MODEL_DIR = os.path.join("models", "fault")
IMG_SIZE = mu.IMG_SIZE
AE_EPOCHS = 40
CLS_EPOCHS = 25
BATCH_SIZE = 16
SEED = 42


def build_autoencoder(img_size):
    inp = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same", strides=2)(inp)   # 64
    x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)     # 32
    x = layers.Conv2D(128, 3, activation="relu", padding="same", strides=2)(x)    # 16
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)  # 32
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)   # 64
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)   # 128
    out = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inp, out, name="autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_classifier(img_size, n_classes):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inp = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def augment(x):
    ds = tf.data.Dataset.from_tensor_slices(x)

    def _aug(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.08)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, img  # autoencoder target == input

    ds = ds.shuffle(512, seed=SEED).map(_aug).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def train_one_category(data_dir, category):
    print(f"\n===== Category: {category} =====")
    out_dir = os.path.join(MODEL_DIR, category)
    os.makedirs(out_dir, exist_ok=True)

    info = mu.load_category_paths(data_dir, category)
    good_paths = info["train_good"] + info["test"].get("good", [])
    if len(good_paths) < 10:
        print(f"[warn] too few 'good' images for {category} ({len(good_paths)}), skipping")
        return

    good_imgs = mu.load_image_batch(good_paths, IMG_SIZE)
    tr_imgs, val_imgs = train_test_split(good_imgs, test_size=0.15, random_state=SEED)

    # ---- 1) Autoencoder for anomaly localization ----
    ae = build_autoencoder(IMG_SIZE)
    train_ds = augment(tr_imgs)
    val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_imgs)).batch(BATCH_SIZE)

    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    ae.fit(train_ds, validation_data=val_ds, epochs=AE_EPOCHS, callbacks=cbs, verbose=2)

    # per-pixel reconstruction-error threshold, calibrated on held-out GOOD images
    recon_val = ae.predict(val_imgs, verbose=0)
    err_val = np.mean((val_imgs - recon_val) ** 2, axis=(1, 2, 3))
    threshold = float(np.mean(err_val) + 3 * np.std(err_val))

    ae.save(os.path.join(out_dir, "autoencoder.keras"))

    # ---- 2) Defect-type classifier ----
    class_names = None
    try:
        X, y, class_names = mu.build_classifier_dataset(data_dir, category, IMG_SIZE)
        if len(class_names) < 2 or len(X) < 15:
            raise ValueError("not enough labeled/varied data for a classifier")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        clf = build_classifier(IMG_SIZE, len(class_names))
        cbs2 = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                               restore_best_weights=True)]
        clf.fit(X_train, y_train, validation_data=(X_val, y_val),
                epochs=CLS_EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs2, verbose=2)
        clf.save(os.path.join(out_dir, "classifier.keras"))
    except Exception as e:
        print(f"[warn] classifier skipped for {category}: {e}")
        clf = None

    meta = {
        "category": category,
        "img_size": IMG_SIZE,
        "recon_error_threshold": threshold,
        "class_names": class_names,
        "has_classifier": clf is not None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {category} models to {out_dir} (threshold={threshold:.5f})")


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_dir = mu.find_mvtec_root(CANDIDATE_DATA_DIRS)
    if data_dir is None:
        raise FileNotFoundError(
            "Could not locate MVTec AD category folders (each needs a "
            "train/good subfolder).\n"
            f"Looked in: {CANDIDATE_DATA_DIRS} (recursively).\n"
            "Fix: either move your MVTec_AD folder so it's at 'MVTec_AD/' or "
            "'data/MVTec_AD/' relative to this script, or edit "
            "CANDIDATE_DATA_DIRS at the top of train_fault_model.py to add "
            "the exact path where your category folders (bottle, cable, ...) live."
        )
    print(f"Using MVTec AD data folder: {os.path.abspath(data_dir)}")

    categories = mu.list_categories(data_dir)
    print(f"Found {len(categories)} categories: {categories}")

    for cat in categories:
        try:
            train_one_category(data_dir, cat)
        except Exception as e:
            print(f"[error] category '{cat}' failed: {e}")

    print("\nAll categories processed. Models saved under:", MODEL_DIR)


if __name__ == "__main__":
    main()
