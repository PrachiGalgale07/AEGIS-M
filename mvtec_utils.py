"""
mvtec_utils.py
--------------
Utilities to discover and load the MVTec AD dataset (ALL categories,
e.g. bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut,
pill, screw, tile, toothbrush, transistor, wood, zipper, ...).

Expected folder layout (standard MVTec AD release, place your real data here):

    data/MVTec_AD/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

Nothing about a specific category is hard-coded: the script discovers
whichever category folders you actually have and trains one model per
category, so it automatically covers "all the files" you provide,
mirroring how all four FD00X CMaps subsets are used together.
"""

import glob
import os

import numpy as np
from PIL import Image

IMG_SIZE = 128  # (IMG_SIZE, IMG_SIZE, 3)


def list_categories(root):
    if not os.path.isdir(root):
        return []
    cats = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "train")):
            cats.append(name)
    return cats


def find_mvtec_root(candidates):
    """Given a list of candidate folder paths, return the first one whose
    immediate children look like MVTec categories (contain train/good), OR
    -- if the categories are nested one level deeper than expected -- the
    correct parent folder found by walking the tree. Returns None if
    nothing usable is found."""
    for c in candidates:
        if not os.path.isdir(c):
            continue
        if list_categories(c):
            return c
        # fall back: search recursively for any ".../<category>/train/good"
        # pattern and infer the correct root from it
        for dirpath, dirnames, _ in os.walk(c):
            if os.path.basename(dirpath) == "train" and "good" in dirnames:
                category_dir = os.path.dirname(dirpath)   # .../<category>
                root = os.path.dirname(category_dir)       # .../<mvtec root>
                if list_categories(root):
                    return root
    return None


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def load_category_paths(root, category):
    """Return dict with train_good, test (defect_type -> paths), gt (defect_type -> mask paths)."""
    cat_dir = os.path.join(root, category)
    info = {
        "train_good": _list_images(os.path.join(cat_dir, "train", "good")),
        "test": {},
        "ground_truth": {},
    }
    test_dir = os.path.join(cat_dir, "test")
    if os.path.isdir(test_dir):
        for defect_type in sorted(os.listdir(test_dir)):
            dpath = os.path.join(test_dir, defect_type)
            if os.path.isdir(dpath):
                info["test"][defect_type] = _list_images(dpath)

    gt_dir = os.path.join(cat_dir, "ground_truth")
    if os.path.isdir(gt_dir):
        for defect_type in sorted(os.listdir(gt_dir)):
            dpath = os.path.join(gt_dir, defect_type)
            if os.path.isdir(dpath):
                info["ground_truth"][defect_type] = _list_images(dpath)
    return info


def load_image(path, size=IMG_SIZE, as_gray=False):
    img = Image.open(path).convert("L" if as_gray else "RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    if as_gray:
        arr = arr[..., None]
    return arr


def load_image_batch(paths, size=IMG_SIZE):
    return np.stack([load_image(p, size) for p in paths], axis=0)


def build_classifier_dataset(root, category, size=IMG_SIZE, include_train_good=True):
    """Build (X, y, class_names) for the defect-type classifier of one category.
    Labels come from the MVTec test/<defect_type>/ and (optionally) train/good/
    folder names. NOTE: because MVTec is designed as an unsupervised anomaly
    benchmark, the labeled test folder is small -- this classifier is a
    supplementary demo head on top of the primary autoencoder-based anomaly
    detector, not a replacement for it.
    """
    info = load_category_paths(root, category)
    paths, labels = [], []

    if include_train_good:
        for p in info["train_good"]:
            paths.append(p)
            labels.append("good")

    for defect_type, plist in info["test"].items():
        for p in plist:
            paths.append(p)
            labels.append(defect_type)

    class_names = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
    X = load_image_batch(paths, size)
    return X, y, class_names
