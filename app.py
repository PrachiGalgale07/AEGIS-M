"""
app.py
------
Battlefield Equipment Predictive Maintenance and Visual Fault Diagnosis - console UI.

Three modes (sidebar):
  1. Sensor Diagnostics  -> upload sensor CSV, get RUL + risk, plus a full
                            degradation trend across the uploaded cycles
  2. Visual Inspection   -> upload one or more component photos, get
                            defect type + localization heatmap per image
  3. Fleet Overview      -> combined session log of every scan run so far,
                            with fleet-wide readiness counts and CSV export

Run:
    streamlit run app.py
"""

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

import cmaps_utils as cu
import mvtec_utils as mu

RUL_MODEL_DIR = os.path.join("models", "rul")
FAULT_MODEL_DIR = os.path.join("models", "fault")

st.set_page_config(
    page_title="AEGIS-M | Equipment Maintenance Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
BG_DEEP = "#12140F"
BG_PANEL = "#1B1E17"
BG_CARD = "#20241B"
BORDER = "#383C2E"
TEXT_PRIMARY = "#DEDFCE"
TEXT_MUTED = "#8B9077"
SIGNAL = "#9CAF6B"     # operational / good
COPPER = "#B8763E"     # accent / headers
AMBER = "#C08A2E"      # degraded
CRITICAL = "#A6493D"   # critical

EQUIPMENT_VOCAB = {
    "Main Battle Tank": "hull / turret assembly",
    "UAV / Drone": "airframe / rotor assembly",
    "Logistics Truck": "chassis / drivetrain",
    "Radar System": "antenna array / housing",
    "Generic Platform": "component",
}

REPAIR_HINTS = {
    "good": "No damage detected. Continue routine inspection schedule.",
    "crack": "Structural crack detected — ground the platform and refer to structural repair team.",
    "scratch": "Cosmetic/surface scratch — monitor; schedule repaint/coating if it deepens.",
    "dent": "Impact dent detected — inspect underlying structure for stress fractures.",
    "bent": "Bent component detected — realign or replace the affected part.",
    "broken": "Component breakage detected — replace part before redeployment.",
    "hole": "Puncture/hole detected — patch or replace immediately, check for internal damage.",
    "contamination": "Contamination detected — clean and inspect for corrosion risk.",
    "corrosion": "Corrosion detected — treat surface and monitor structural integrity.",
    "cut": "Cut/tear detected — repair or replace affected material.",
    "missing": "Missing component detected — reinstall/replace part before use.",
    "misalignment": "Misalignment detected — recalibrate/realign the component.",
    "print": "Print/marking defect detected — inspect for functional impact.",
    "thread": "Thread damage detected — inspect fastening integrity, replace if compromised.",
}


def repair_hint(defect_label):
    label = defect_label.lower()
    if label == "good":
        return REPAIR_HINTS["good"]
    for key, hint in REPAIR_HINTS.items():
        if key in label:
            return hint
    return "Anomaly detected — recommend manual inspection by maintenance crew."


# --------------------------------------------------------------------------
# CSS — dark, muted defense-console look (no bright/neon accents)
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {TEXT_PRIMARY};
        }}
        .stApp {{
            background-color: {BG_DEEP};
        }}
        [data-testid="stSidebar"] {{
            background-color: {BG_PANEL};
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0);
        }}

        /* ---- console header ---- */
        .console-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            border-bottom: 1px solid {BORDER};
            padding-bottom: 10px;
            margin-bottom: 4px;
        }}
        .console-title {{
            font-family: 'IBM Plex Sans Condensed', sans-serif;
            font-weight: 700;
            font-size: 1.6rem;
            letter-spacing: 0.04em;
            color: {TEXT_PRIMARY};
            text-transform: uppercase;
        }}
        .console-title span {{ color: {COPPER}; }}
        .console-subtitle {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.03em;
        }}

        /* ---- telemetry strip (signature element) ---- */
        .telemetry-strip {{
            display: flex;
            gap: 28px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 8px 2px 18px 2px;
        }}
        .telemetry-strip b {{ color: {TEXT_PRIMARY}; font-weight: 500; }}
        .dot {{
            display: inline-block; width: 7px; height: 7px; border-radius: 50%;
            background-color: {SIGNAL}; margin-right: 6px;
        }}

        /* ---- section labels ---- */
        .section-label {{
            font-family: 'IBM Plex Sans Condensed', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {COPPER};
            border-bottom: 1px solid {BORDER};
            padding-bottom: 6px;
            margin: 18px 0 12px 0;
        }}

        /* ---- cards ---- */
        .console-card {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 3px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }}

        /* ---- badges ---- */
        .badge {{
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 2px;
            border: 1px solid;
        }}
        .badge-operational {{ color: {SIGNAL}; border-color: {SIGNAL}; background: rgba(156,175,107,0.08); }}
        .badge-degraded {{ color: {AMBER}; border-color: {AMBER}; background: rgba(192,138,46,0.08); }}
        .badge-critical {{ color: {CRITICAL}; border-color: {CRITICAL}; background: rgba(166,73,61,0.10); }}

        /* ---- streamlit widget restyling ---- */
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace;
            color: {TEXT_PRIMARY};
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'IBM Plex Sans Condensed', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            color: {TEXT_MUTED};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'IBM Plex Sans Condensed', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.85rem;
            color: {TEXT_MUTED};
        }}
        .stTabs [aria-selected="true"] {{
            color: {SIGNAL} !important;
        }}
        .stButton>button {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER};
            border-radius: 3px;
            font-family: 'IBM Plex Sans Condensed', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.8rem;
        }}
        .stButton>button:hover {{
            border-color: {SIGNAL};
            color: {SIGNAL};
        }}
        hr {{ border-color: {BORDER}; }}
        .footnote {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem;
            color: {TEXT_MUTED};
            margin-top: 24px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(status):
    cls = {"OPERATIONAL": "badge-operational", "DEGRADED": "badge-degraded",
           "CRITICAL": "badge-critical"}.get(status, "badge-degraded")
    return f'<span class="badge {cls}">{status}</span>'


def classify_status(risk_prob, rul_val, risk_threshold):
    if risk_prob > 0.7 or rul_val < risk_threshold:
        return "CRITICAL"
    if risk_prob > 0.4 or rul_val < risk_threshold * 2:
        return "DEGRADED"
    return "OPERATIONAL"


def gauge_figure(value, title, value_range, steps, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"family": "IBM Plex Mono", "size": 30, "color": TEXT_PRIMARY}},
        title={"text": title, "font": {"family": "IBM Plex Sans Condensed", "size": 13, "color": TEXT_MUTED}},
        gauge={
            "axis": {"range": value_range, "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED, "size": 9}},
            "bar": {"color": COPPER, "thickness": 0.3},
            "bgcolor": BG_CARD,
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": steps,
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(color=TEXT_PRIMARY),
    )
    return fig


# --------------------------------------------------------------------------
# Cached model / artifact loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_rul_artifacts():
    model_path = os.path.join(RUL_MODEL_DIR, "rul_model.keras")
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    kmeans, sensor_scalers, op_scaler, config = cu.load_preprocessing(RUL_MODEL_DIR)
    return {
        "model": model, "kmeans": kmeans, "sensor_scalers": sensor_scalers,
        "op_scaler": op_scaler, "config": config,
    }


@st.cache_resource
def list_fault_categories():
    if not os.path.isdir(FAULT_MODEL_DIR):
        return []
    return sorted(
        d for d in os.listdir(FAULT_MODEL_DIR)
        if os.path.isdir(os.path.join(FAULT_MODEL_DIR, d))
    )


@st.cache_resource
def load_fault_artifacts(category):
    cat_dir = os.path.join(FAULT_MODEL_DIR, category)
    with open(os.path.join(cat_dir, "meta.json")) as f:
        meta = json.load(f)
    ae = tf.keras.models.load_model(os.path.join(cat_dir, "autoencoder.keras"))
    clf = None
    clf_path = os.path.join(cat_dir, "classifier.keras")
    if meta.get("has_classifier") and os.path.exists(clf_path):
        clf = tf.keras.models.load_model(clf_path)
    return {"autoencoder": ae, "classifier": clf, "meta": meta}


# --------------------------------------------------------------------------
# Session state (in-memory scan log, resets each session)
# --------------------------------------------------------------------------
def init_state():
    if "rul_history" not in st.session_state:
        st.session_state.rul_history = []
    if "fault_history" not in st.session_state:
        st.session_state.fault_history = []


# --------------------------------------------------------------------------
# Mode 1: Sensor Diagnostics (RUL + risk + trend)
# --------------------------------------------------------------------------
def compute_rul_trend(features, raw, window_size, model, max_points=60):
    """Slide a window across the WHOLE uploaded history and return a
    predicted-RUL / risk curve, not just a single end-point prediction."""
    n_original = len(features)
    pad = max(0, window_size - n_original)
    padded = np.vstack([np.repeat(features[:1], pad, axis=0), features]) if pad else features
    n_padded = len(padded)

    step = max(1, (n_padded - window_size) // max_points) if n_padded > window_size else 1
    starts = list(range(0, n_padded - window_size + 1, step))
    if starts[-1] != n_padded - window_size:
        starts.append(n_padded - window_size)

    batch = np.stack([padded[s:s + window_size] for s in starts]).astype(np.float32)
    rul_preds, risk_preds = model.predict(batch, verbose=0)

    cycles = []
    has_cycle_col = "cycle" in raw.columns
    for s in starts:
        original_idx = s + window_size - 1 - pad
        if has_cycle_col:
            cycles.append(int(raw["cycle"].iloc[original_idx]))
        else:
            cycles.append(original_idx + 1)

    return cycles, rul_preds.flatten(), risk_preds.flatten()


def rul_mode(equipment_type):
    st.markdown('<div class="section-label">Sensor Diagnostics</div>', unsafe_allow_html=True)
    st.caption(
        "Model trained jointly on all four NASA C-MAPSS subsets (FD001-FD004). "
        "Upload the most recent sensor cycles for one platform to forecast Remaining "
        "Useful Life and failure risk."
    )

    artifacts = load_rul_artifacts()
    if artifacts is None:
        st.warning(
            f"No trained RUL model found at `{RUL_MODEL_DIR}`. Run `python train_rul_model.py` first."
        )
        return

    config = artifacts["config"]
    window_size = config["window_size"]
    needed_cols = config["op_cols"] + config["sensor_cols"]

    with st.expander("Expected CSV format"):
        st.write(
            f"Columns required: `cycle`, {', '.join(needed_cols)}. "
            f"At least 1 row; ideally ≥ {window_size} rows (most recent cycles, in order). "
            "Shorter histories are automatically padded."
        )

    uploaded = st.file_uploader("Upload sensor CSV", type=["csv", "txt"], key="rul_upload")
    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as e:
        st.error(f"Could not parse file: {e}")
        return

    missing = [c for c in needed_cols if c not in raw.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    raw = raw.sort_values("cycle").reset_index(drop=True) if "cycle" in raw.columns else raw
    features, _ = cu.transform_features(
        raw, artifacts["kmeans"], artifacts["sensor_scalers"], artifacts["op_scaler"],
        n_clusters=config["n_op_clusters"],
    )

    cycles, rul_series, risk_series = compute_rul_trend(features, raw, window_size, artifacts["model"])
    rul_val, risk_val = float(rul_series[-1]), float(risk_series[-1])
    status = classify_status(risk_val, rul_val, config["risk_threshold"])

    st.markdown('<div class="console-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.plotly_chart(
            gauge_figure(
                rul_val, "PREDICTED RUL (CYCLES)", [0, max(125, rul_val * 1.2)],
                steps=[
                    {"range": [0, config["risk_threshold"]], "color": "rgba(166,73,61,0.35)"},
                    {"range": [config["risk_threshold"], config["risk_threshold"] * 2], "color": "rgba(192,138,46,0.30)"},
                    {"range": [config["risk_threshold"] * 2, max(125, rul_val * 1.2)], "color": "rgba(156,175,107,0.25)"},
                ],
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c2:
        st.plotly_chart(
            gauge_figure(
                risk_val * 100, "FAILURE RISK", [0, 100], suffix="%",
                steps=[
                    {"range": [0, 40], "color": "rgba(156,175,107,0.25)"},
                    {"range": [40, 70], "color": "rgba(192,138,46,0.30)"},
                    {"range": [70, 100], "color": "rgba(166,73,61,0.35)"},
                ],
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Platform status**<br>{badge_html(status)}", unsafe_allow_html=True)
        st.markdown(f"<br>**Assembly:** {EQUIPMENT_VOCAB[equipment_type]}", unsafe_allow_html=True)
        if status == "CRITICAL":
            st.markdown(f"Ground the {equipment_type.lower()} and schedule immediate maintenance.")
        elif status == "DEGRADED":
            st.markdown("Plan maintenance within the next mission cycle.")
        else:
            st.markdown("No action required — continue routine monitoring.")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(cycles) > 1:
        st.markdown('<div class="section-label">Degradation Trend</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles, y=rul_series, name="Predicted RUL", mode="lines+markers",
            line=dict(color=SIGNAL, width=2), marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=cycles, y=[r * 100 for r in risk_series], name="Risk probability (%)",
            mode="lines", yaxis="y2", line=dict(color=CRITICAL, width=1.5, dash="dot"),
        ))
        fig.add_hline(y=config["risk_threshold"], line_dash="dash", line_color=AMBER,
                       annotation_text="risk threshold", annotation_font_color=TEXT_MUTED)
        fig.update_layout(
            xaxis=dict(title="Cycle", color=TEXT_MUTED, gridcolor=BORDER),
            yaxis=dict(title="Predicted RUL (cycles)", color=TEXT_MUTED, gridcolor=BORDER),
            yaxis2=dict(title="Risk probability (%)", overlaying="y", side="right", range=[0, 100], color=TEXT_MUTED),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color=TEXT_PRIMARY, size=11),
            legend=dict(orientation="h", y=1.18, font=dict(color=TEXT_MUTED)),
            margin=dict(l=10, r=10, t=30, b=10), height=320,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.session_state.rul_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "equipment_type": equipment_type,
        "source_file": uploaded.name,
        "predicted_rul": round(rul_val, 1),
        "risk_probability": round(risk_val, 3),
        "status": status,
    })


# --------------------------------------------------------------------------
# Mode 2: Visual Inspection (batch fault diagnosis + localization)
# --------------------------------------------------------------------------
def make_heatmap_overlay(orig_img, recon_img, threshold):
    err_map = np.mean((orig_img - recon_img) ** 2, axis=-1)
    norm = err_map / (err_map.max() + 1e-8)
    colored = cm.jet(norm)[..., :3]
    overlay = 0.55 * orig_img + 0.45 * colored
    overlay = np.clip(overlay, 0, 1)
    mean_err = float(err_map.mean())
    return overlay, mean_err, mean_err > threshold


def fault_mode(equipment_type):
    st.markdown('<div class="section-label">Visual Inspection</div>', unsafe_allow_html=True)
    st.caption(
        "Per-category autoencoder localizes damaged regions; a classifier names the "
        "defect type. Upload one or more photos of the same component category."
    )

    categories = list_fault_categories()
    if not categories:
        st.warning(
            f"No trained fault models found at `{FAULT_MODEL_DIR}`. Run `python train_fault_model.py` first."
        )
        return

    category = st.selectbox("Component category", categories)
    artifacts = load_fault_artifacts(category)
    meta = artifacts["meta"]
    img_size = meta["img_size"]

    uploaded_files = st.file_uploader(
        "Upload component image(s)", type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True, key="fault_upload",
    )
    if not uploaded_files:
        return

    results = []
    for uploaded in uploaded_files:
        pil_img = Image.open(uploaded).convert("RGB")
        resized = pil_img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(resized).astype("float32") / 255.0

        recon = artifacts["autoencoder"].predict(arr[None, ...], verbose=0)[0]
        overlay, mean_err, is_defective = make_heatmap_overlay(arr, recon, meta["recon_error_threshold"])

        label, confidence = ("good", 1.0) if not is_defective else ("anomaly", None)
        if artifacts["classifier"] is not None:
            probs = artifacts["classifier"].predict(arr[None, ...], verbose=0)[0]
            idx = int(np.argmax(probs))
            label = meta["class_names"][idx]
            confidence = float(probs[idx])

        defective_final = is_defective or label != "good"
        results.append({
            "filename": uploaded.name, "pil_img": pil_img, "overlay": overlay,
            "label": label, "confidence": confidence, "mean_err": mean_err,
            "defective": defective_final,
        })

    st.markdown('<div class="section-label">Inspection Results</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            st.markdown('<div class="console-card">', unsafe_allow_html=True)
            st.image(r["pil_img"], use_container_width=True)
            status = "CRITICAL" if r["defective"] else "OPERATIONAL"
            st.markdown(badge_html(status), unsafe_allow_html=True)
            st.markdown(f"**{r['label']}**", unsafe_allow_html=True)
            if r["confidence"] is not None:
                st.caption(f"confidence {r['confidence']*100:.1f}% · error {r['mean_err']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    inspect_name = st.selectbox("Inspect a result in detail", [r["filename"] for r in results])
    r = next(x for x in results if x["filename"] == inspect_name)
    c1, c2 = st.columns(2)
    with c1:
        st.image(r["pil_img"], caption="Original", use_container_width=True)
    with c2:
        st.image(r["overlay"], caption="Defect localization heatmap", use_container_width=True)
    if r["defective"]:
        st.error(f"Recommended action for this {EQUIPMENT_VOCAB[equipment_type]}: {repair_hint(r['label'])}")
    else:
        st.success(repair_hint("good"))

    if len(results) > 1:
        st.markdown('<div class="section-label">Batch Summary</div>', unsafe_allow_html=True)
        counts = pd.Series([r["label"] for r in results]).value_counts()
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(), y=counts.values.tolist(),
            marker_color=[SIGNAL if lbl == "good" else CRITICAL for lbl in counts.index],
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color=TEXT_PRIMARY, size=11),
            xaxis=dict(color=TEXT_MUTED, gridcolor=BORDER),
            yaxis=dict(title="Count", color=TEXT_MUTED, gridcolor=BORDER),
            height=260, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    for r in results:
        st.session_state.fault_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "equipment_type": equipment_type,
            "category": category,
            "filename": r["filename"],
            "defect_type": r["label"],
            "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None,
            "status": "CRITICAL" if r["defective"] else "OPERATIONAL",
        })


# --------------------------------------------------------------------------
# Mode 3: Fleet Overview (combined session log)
# --------------------------------------------------------------------------
def fleet_overview_mode():
    st.markdown('<div class="section-label">Fleet Overview</div>', unsafe_allow_html=True)
    st.caption("Combined readiness log for every scan run so far this session.")

    rul_df = pd.DataFrame(st.session_state.rul_history)
    fault_df = pd.DataFrame(st.session_state.fault_history)

    if rul_df.empty and fault_df.empty:
        st.info("No diagnostics run yet this session. Run a scan in Sensor or Visual "
                "Inspection mode to populate the fleet log.")
        return

    statuses = pd.concat([
        rul_df["status"] if not rul_df.empty else pd.Series(dtype=str),
        fault_df["status"] if not fault_df.empty else pd.Series(dtype=str),
    ])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Scans", len(statuses))
    k2.metric("Operational", int((statuses == "OPERATIONAL").sum()))
    k3.metric("Degraded", int((statuses == "DEGRADED").sum()))
    k4.metric("Critical", int((statuses == "CRITICAL").sum()))

    if not rul_df.empty:
        st.markdown('<div class="section-label">Sensor Diagnostics Log</div>', unsafe_allow_html=True)
        st.dataframe(rul_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Download sensor log (CSV)", rul_df.to_csv(index=False).encode("utf-8"),
            file_name="sensor_diagnostics_log.csv", mime="text/csv",
        )

    if not fault_df.empty:
        st.markdown('<div class="section-label">Visual Inspection Log</div>', unsafe_allow_html=True)
        st.dataframe(fault_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Download inspection log (CSV)", fault_df.to_csv(index=False).encode("utf-8"),
            file_name="visual_inspection_log.csv", mime="text/csv",
        )


# --------------------------------------------------------------------------
def main():
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown('<div class="console-title">AEGIS<span>-M</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="console-subtitle">MAINTENANCE &amp; FAULT DIAGNOSIS CONSOLE</div>', unsafe_allow_html=True)
        st.markdown("---")
        equipment_type = st.selectbox("Platform type", list(EQUIPMENT_VOCAB.keys()))
        mode = st.radio("Mode", ["Sensor Diagnostics", "Visual Inspection", "Fleet Overview"])
        st.markdown("---")
        st.caption(
            "Sensor-based RUL/risk forecasting (C-MAPSS, all subsets) + "
            "image-based fault localization (MVTec AD, all categories)."
        )

    st.markdown(
        f'<div class="console-header">'
        f'<div class="console-title">Equipment Readiness Console</div>'
        f'<div class="console-subtitle">{equipment_type.upper()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    scans_run = len(st.session_state.rul_history) + len(st.session_state.fault_history)
    st.markdown(
        f'<div class="telemetry-strip">'
        f'<span><span class="dot"></span><b>SESSION ACTIVE</b></span>'
        f'<span>DATE&nbsp;<b>{datetime.now().strftime("%Y-%m-%d")}</b></span>'
        f'<span>SCANS THIS SESSION&nbsp;<b>{scans_run}</b></span>'
        f'<span>MODE&nbsp;<b>{mode.upper()}</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if mode == "Sensor Diagnostics":
        rul_mode(equipment_type)
    elif mode == "Visual Inspection":
        fault_mode(equipment_type)
    else:
        fleet_overview_mode()

    st.markdown(
        '<div class="footnote">Prototype system for academic demonstration — '
        'not certified for operational deployment.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
