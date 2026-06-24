"""
backend/compare.py — Multi-model comparison.

Trains multiple models on the same dataset and compares:
- Performance metrics side by side
- Global SHAP feature importance per model
- Agreement score: how much do models agree on which features matter?

Design:
- Each model is trained independently on the same X_train/y_train
- SHAP computed on the same X_test sample for fair comparison
- Agreement = Spearman rank correlation of feature importance vectors
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
from scipy.stats import spearmanr

from backend.model import train_model, get_model_options
from backend.shap_explainer import get_shap_explainer, compute_shap_values, global_shap_summary


def run_comparison(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    X_shap_background: pd.DataFrame,
    feature_names: list,
    task: str,
    is_imbalanced: bool,
    minority_ratio: float,
    models_to_compare: list,
) -> dict:
    """
    Train and evaluate multiple models. Return comparison results.

    Returns dict with:
        results    — list of per-model dicts (metrics, shap summary, model object)
        feature_names — list of feature names
    """
    results = []

    for model_name in models_to_compare:
        try:
            model, strategy, X_used = train_model(
                X_train, y_train, task, model_name, is_imbalanced, minority_ratio
            )

            y_pred = model.predict(X_test)

            # Metrics
            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                try:
                    auc = round(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), 4)
                except Exception:
                    auc = "N/A"
                metrics = {"Accuracy": round(acc, 4), "F1": round(f1, 4), "ROC-AUC": auc}
            else:
                r2   = r2_score(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = float(np.sqrt(np.mean((np.array(y_test) - np.array(y_pred))**2)))
                metrics = {"R²": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}

            # SHAP
            explainer  = get_shap_explainer(model, X_shap_background)
            shap_vals  = compute_shap_values(explainer, X_test)
            summary    = global_shap_summary(shap_vals, feature_names, top_n=len(feature_names))

            results.append({
                "model_name":   model_name,
                "model":        model,
                "metrics":      metrics,
                "shap_summary": summary,
                "strategy":     strategy,
            })

        except Exception as e:
            results.append({
                "model_name": model_name,
                "error":      str(e),
            })

    return {"results": results, "feature_names": feature_names}


def plot_metrics_comparison(results: list, task: str) -> plt.Figure:
    """Bar chart comparing metrics across models."""
    valid   = [r for r in results if "error" not in r]
    if not valid:
        return None

    model_names = [r["model_name"] for r in valid]
    metric_keys = list(valid[0]["metrics"].keys())

    # Filter out N/A
    numeric_keys = [k for k in metric_keys
                    if all(r["metrics"].get(k, "N/A") != "N/A" for r in valid)]

    n_metrics = len(numeric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = ["#667eea", "#28a745", "#dc3545", "#f0a500", "#17a2b8"]

    for ax, metric in zip(axes, numeric_keys):
        vals = [float(r["metrics"][metric]) for r in valid]
        bars = ax.bar(model_names, vals,
                      color=colors[:len(model_names)], width=0.5)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, min(1.15, max(vals) * 1.2) if max(vals) <= 1 else max(vals) * 1.2)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_shap_comparison(results: list, feature_names: list, top_n: int = 8) -> plt.Figure:
    """
    Side-by-side horizontal bar charts of global SHAP importance per model.
    Uses mean |SHAP| so all bars go right — easier to compare across models.
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    n_models = len(valid)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7), sharey=False)
    if n_models == 1:
        axes = [axes]

    colors = ["#667eea", "#28a745", "#dc3545", "#f0a500", "#17a2b8"]

    for ax, result, color in zip(axes, valid, colors):
        summary  = result["shap_summary"]
        top_idx  = summary["top_idx"][:top_n]
        top_feats= [feature_names[i] for i in top_idx]
        top_abs  = [float(summary["mean_abs"][i]) for i in top_idx]
        top_sign = [float(summary["mean_sign"][i]) for i in top_idx]
        total    = summary["total_abs"]
        bar_vals = [a if s >= 0 else -a for a, s in zip(top_abs, top_sign)]
        bar_colors = ["#28a745" if s >= 0 else "#dc3545" for s in top_sign]
        max_abs  = max(top_abs) if top_abs else 1

        ax.barh(range(len(top_feats)), bar_vals, color=bar_colors, height=0.6)
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-max_abs * 1.3, max_abs * 1.3)

        for i, (bv, av) in enumerate(zip(bar_vals, top_abs)):
            pct    = (av / total * 100) if total > 0 else 0
            offset = max_abs * 0.03
            if bv >= 0:
                ax.text(bv + offset, i, f"{pct:.1f}%", va="center", ha="left",  fontsize=8)
            else:
                ax.text(bv - offset, i, f"{pct:.1f}%", va="center", ha="right", fontsize=8)

        ax.set_title(result["model_name"], fontsize=11, fontweight="bold", color=color)
        ax.invert_yaxis()

    plt.suptitle("SHAP Feature Importance — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def compute_agreement(results: list, feature_names: list) -> pd.DataFrame:
    """
    Compute Spearman rank correlation between feature importance vectors of all model pairs.
    Returns a DataFrame (model x model) with correlation values.
    High correlation = models agree on which features matter.
    """
    valid = [r for r in results if "error" not in r]
    if len(valid) < 2:
        return None

    model_names = [r["model_name"] for r in valid]
    importance_vecs = []

    for r in valid:
        summary = r["shap_summary"]
        mean_abs = summary["mean_abs"]
        # Full vector across all features
        vec = np.array([float(mean_abs[i]) for i in range(len(feature_names))])
        importance_vecs.append(vec)

    n = len(valid)
    corr_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i+1, n):
            corr, _ = spearmanr(importance_vecs[i], importance_vecs[j])
            corr_matrix[i][j] = round(corr, 3)
            corr_matrix[j][i] = round(corr, 3)

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)
