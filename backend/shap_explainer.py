"""
backend/shap_explainer.py — Production SHAP computation.

Real-world fixes vs prototype:
1. TreeExplainer used explicitly for tree models (not generic shap.Explainer).
2. LinearExplainer for linear models.
3. KernelExplainer only for SVM — with hard cap of 100 background rows.
4. SHAP background is passed from preprocess.py (stratified sample of train set).
5. Large test sets → SHAP computed on a sample for speed, then extrapolated
   for global importance. Local SHAP always computed on exact row.
6. 3D SHAP output (multi-class) handled correctly throughout.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap

# Max test rows to compute global SHAP on (speed vs accuracy tradeoff)
MAX_GLOBAL_SHAP_ROWS = 1000

TREE_MODELS = {
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
}

LINEAR_MODELS = {
    "LinearRegression", "LogisticRegression", "Ridge",
    "Lasso", "ElasticNet",
}


def get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Select the correct SHAP explainer for the model type.

    - Tree models  → TreeExplainer (exact, fast)
    - Linear       → LinearExplainer (exact, fast)
    - SVM / other  → KernelExplainer (slow, background capped at 100 rows)
    """
    model_class = type(model).__name__

    # Unwrap CalibratedClassifierCV to check inner estimator
    inner_class = model_class
    if model_class == "CalibratedClassifierCV":
        try:
            inner_class = type(model.estimator).__name__
        except AttributeError:
            inner_class = type(model.base_estimator).__name__

    if model_class in TREE_MODELS or inner_class in TREE_MODELS:
        # GradientBoostingClassifier: force feature_perturbation="tree_path_dependent"
        # to ensure output is always a list [class0, class1], never a raw 2D array
        # that could be class-0 oriented (which flips all bar directions).
        gb_classes = {"GradientBoostingClassifier"}
        target_class = model_class if model_class != "CalibratedClassifierCV" else inner_class
        if target_class in gb_classes:
            return shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent",
            )
        return shap.TreeExplainer(model)

    if model_class in LINEAR_MODELS or inner_class in LINEAR_MODELS:
        return shap.LinearExplainer(model, X_background)

    # KernelExplainer fallback (SVM etc.) — strict background cap
    bg = shap.sample(X_background, min(100, len(X_background)), random_state=42)
    return shap.KernelExplainer(model.predict, bg)


def _normalize_shap_output(raw, arr: np.ndarray, target_class: int = None,
                           explainer=None) -> np.ndarray:
    """
    Always return 2D array (n_samples, n_features) oriented to class-1 (positive class).

    SHAP output shapes vary by version and model type:

      Older SHAP + RF/GB  → list of 2 arrays [class0_shap, class1_shap]
      Newer SHAP + RF/GB  → single 2D array, but may be class-0 oriented
                            (all signs flipped vs what we want)
      XGBClassifier       → 2D array, already class-1 oriented
      3D array            → (n_samples, n_features, n_classes)

    Root cause of the color-flip bug: newer SHAP TreeExplainer on RandomForest
    returns a plain 2D array that is class-0 oriented (signs all inverted).
    Fix: when we get a 2D array from a binary classifier, use expected_value
    to detect orientation and flip if needed.
    """
    # ── Case 1: list of arrays (older sklearn TreeExplainer for RF, GB) ──────
    if isinstance(raw, list):
        n_classes = len(raw)
        if n_classes == 2:
            out = np.array(raw[1])          # always class-1
        elif target_class is not None and target_class < n_classes:
            out = np.array(raw[target_class])
        else:
            stacked = np.stack([np.array(r) for r in raw], axis=0)
            best = int(np.argmax(np.abs(stacked).mean(axis=(1, 2))))
            out = stacked[best]
        return out if out.ndim == 2 else out.reshape(1, -1)

    # ── Case 2: 3D array (n_samples, n_features, n_classes) ──────────────────
    if arr.ndim == 3:
        n_classes = arr.shape[2]
        if n_classes == 2:
            return arr[:, :, 1]
        if target_class is not None and target_class < n_classes:
            return arr[:, :, target_class]
        best = int(np.argmax(np.abs(arr).mean(axis=(0, 1))))
        return arr[:, :, best]

    # ── Case 3: plain 2D array ────────────────────────────────────────────────
    # May be class-0 oriented (newer SHAP + RF). Detect via expected_value.
    # If the explainer has two expected values (binary classifier) and the
    # array's mean sign is inverted relative to expected_value[1], flip it.
    if arr.ndim == 2 and explainer is not None:
        ev = getattr(explainer, "expected_value", None)
        if ev is not None:
            ev_arr = np.atleast_1d(ev)
            if len(ev_arr) == 2:
                ev0, ev1 = float(ev_arr[0]), float(ev_arr[1])
                arr_mean = float(arr.mean())
                # If class-1 base rate > class-0 base rate, class-1 SHAP values
                # should have positive mean on average. If arr_mean < 0, flipped.
                if ev1 > ev0 and arr_mean < 0:
                    return -arr
                if ev0 > ev1 and arr_mean > 0:
                    return -arr

    return arr


def compute_shap_values(explainer, X: pd.DataFrame,
                        predicted_class: int = None) -> np.ndarray:
    """
    Compute SHAP values for a dataset.
    For large datasets, computes on a sample for global explainability.
    Always returns 2D (n_samples, n_features).

    predicted_class: for multi-class, which class index to extract SHAP for.
                     None → binary default (class 1) or highest-mean-abs class.
    """
    if len(X) > MAX_GLOBAL_SHAP_ROWS:
        X = X.sample(n=MAX_GLOBAL_SHAP_ROWS, random_state=42).reset_index(drop=True)

    raw = explainer.shap_values(X)
    return _normalize_shap_output(raw, np.array(raw),
                                  target_class=predicted_class, explainer=explainer)


def compute_single_row_shap(explainer, X_single: pd.DataFrame,
                             predicted_class: int = None) -> np.ndarray:
    """
    Compute SHAP for exactly one row. Returns 1D array.
    Always uses the full row — never sampled.

    predicted_class: for multi-class, extract SHAP for this class index.
    """
    raw = explainer.shap_values(X_single)
    arr = _normalize_shap_output(raw, np.array(raw),
                                 target_class=predicted_class, explainer=explainer)
    return arr[0]


def get_local_shap(shap_vals: np.ndarray, row_idx: int) -> np.ndarray:
    """Extract 1D SHAP vector for a single row from precomputed values."""
    return shap_vals[row_idx]


# ── Global summary ────────────────────────────────────────────────────────────

def global_shap_summary(shap_vals: np.ndarray, feature_names: list, top_n: int = 10) -> dict:
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim != 2:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_sign = shap_vals.mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n].tolist()

    return {
        "top_idx": top_idx,
        "mean_abs": mean_abs,
        "mean_sign": mean_sign,
        "top_features": [feature_names[i] for i in top_idx],
        "total_abs": float(mean_abs.sum()),
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_global_shap(summary: dict, feature_names: list, positive_class: str = "prediction") -> plt.Figure:
    top_idx      = summary["top_idx"]
    top_features = [feature_names[i] for i in top_idx]
    top_sign     = [float(summary["mean_sign"][i]) for i in top_idx]
    top_abs      = [float(summary["mean_abs"][i])  for i in top_idx]
    total        = summary["total_abs"]

    # Direction: positive → draw RIGHT (green), negative → draw LEFT (red)
    # Bar length = mean_abs so the size always matches the % label
    bar_vals = [abs_v if s >= 0 else -abs_v for abs_v, s in zip(top_abs, top_sign)]
    colors   = ["#28a745" if s >= 0 else "#dc3545" for s in top_sign]
    max_abs  = max(top_abs) if top_abs else 1

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(range(len(top_features)), bar_vals, color=colors, height=0.6)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.axvline(0, color="black", linewidth=1.2)

    # % label placed just past the tip of each bar
    for i, (bar_v, abs_v) in enumerate(zip(bar_vals, top_abs)):
        pct    = (abs_v / total * 100) if total > 0 else 0
        offset = max_abs * 0.02
        if bar_v >= 0:
            ax.text(bar_v + offset, i, f"{pct:.1f}%", va="center", ha="left",  fontsize=9, color="#333")
        else:
            ax.text(bar_v - offset, i, f"{pct:.1f}%", va="center", ha="right", fontsize=9, color="#333")

    ax.set_xlabel(
        f"← DECREASES P({positive_class})  |  INCREASES P({positive_class}) →   (bar length = importance)",
        fontsize=11,
    )
    ax.set_title(
        "Global Feature Importance via SHAP\n(averaged over test samples)",
        fontsize=14, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)

    legend_elements = [
        Patch(facecolor="#28a745", label=f"Green → INCREASES P({positive_class})"),
        Patch(facecolor="#dc3545", label=f"Red   → DECREASES P({positive_class})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_local_shap(local_shap: np.ndarray, feature_names: list, positive_class: str = "prediction") -> plt.Figure:
    local_shap = np.array(local_shap).flatten()
    top_n      = min(10, len(feature_names))
    abs_shap   = np.abs(local_shap)
    top_idx    = np.argsort(abs_shap)[-top_n:]
    top_feats  = [feature_names[i] for i in top_idx]
    top_vals   = np.array([local_shap[i] for i in top_idx])
    top_abs    = np.abs(top_vals)
    total_abs  = top_abs.sum()
    colors     = ["#28a745" if v >= 0 else "#dc3545" for v in top_vals]
    max_abs    = top_abs.max() if len(top_abs) > 0 else 1

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top_idx)), top_vals, color=colors, height=0.6)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_feats, fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)

    # % label past each bar tip
    for i, (val, abs_v) in enumerate(zip(top_vals, top_abs)):
        pct    = (abs_v / total_abs * 100) if total_abs > 0 else 0
        offset = max_abs * 0.02
        if val >= 0:
            ax.text(val + offset, i, f"{pct:.1f}%", va="center", ha="left",  fontsize=9, color="#333")
        else:
            ax.text(val - offset, i, f"{pct:.1f}%", va="center", ha="right", fontsize=9, color="#333")

    ax.set_xlabel(
        f"← DECREASES P({positive_class})  |  INCREASES P({positive_class}) →",
        fontsize=11,
    )
    ax.set_title(
        "Local Feature Contributions\n(specific to the selected row)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    plt.tight_layout()
    return fig