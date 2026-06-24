"""
backend/report.py — PDF report generation using ReportLab.

Generates a client-ready PDF containing:
  - Cover page with model info and dataset summary
  - Model performance metrics (plain English interpretation)
  - Global SHAP chart + plain English feature explanations
  - Local SHAP chart + plain English row explanation (if provided)
  - Data quality notes (dropped columns, imbalance handling)

Plain English mode: technical terms replaced with business-friendly language
so non-technical clients can understand the report without ML knowledge.
"""

import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ── Colour palette ────────────────────────────────────────────────────────────

PRIMARY   = colors.HexColor("#667eea")
GREEN     = colors.HexColor("#28a745")
RED       = colors.HexColor("#dc3545")
DARK      = colors.HexColor("#2c3e50")
GREY      = colors.HexColor("#6c757d")
LIGHT_BG  = colors.HexColor("#f8f9fa")
WHITE     = colors.white


# ── Plain-English metric interpretation ───────────────────────────────────────

def _interpret_accuracy(acc: float, is_imbalanced: bool) -> str:
    if is_imbalanced:
        return (
            f"The model correctly classified {acc:.1%} of test cases. "
            "However, because the dataset is imbalanced, accuracy alone can be misleading — "
            "check F1 and ROC-AUC for a more complete picture."
        )
    if acc >= 0.90:
        return f"Excellent — the model correctly predicted {acc:.1%} of test cases."
    elif acc >= 0.75:
        return f"Good — the model correctly predicted {acc:.1%} of test cases."
    else:
        return (
            f"The model correctly predicted {acc:.1%} of test cases. "
            "Consider trying a different model or reviewing your features."
        )


def _interpret_f1(f1: float) -> str:
    if f1 >= 0.85:
        return f"Strong F1 score of {f1:.3f} — the model handles both classes well."
    elif f1 >= 0.65:
        return f"Moderate F1 score of {f1:.3f} — some room for improvement."
    else:
        return (
            f"Low F1 score of {f1:.3f} — the model struggles with the minority class. "
            "Consider collecting more data or tuning the model."
        )


def _interpret_auc(auc) -> str:
    if auc == "N/A":
        return "ROC-AUC not available for this model."
    auc = float(auc)
    if auc >= 0.90:
        return f"Excellent ROC-AUC of {auc:.3f} — the model separates classes very well."
    elif auc >= 0.75:
        return f"Good ROC-AUC of {auc:.3f} — reasonable class separation."
    else:
        return f"Low ROC-AUC of {auc:.3f} — model may need more feature engineering."


def _interpret_r2(r2: float) -> str:
    if r2 >= 0.85:
        return f"Strong R² of {r2:.3f} — the model explains {r2:.1%} of variance in the target."
    elif r2 >= 0.60:
        return f"Moderate R² of {r2:.3f} — the model captures major trends."
    else:
        return f"Low R² of {r2:.3f} — predictions may not be reliable."


# ── Chart rendering ───────────────────────────────────────────────────────────

def _fig_to_image(fig: plt.Figure, width_cm: float = 16) -> Image:
    """Convert a matplotlib figure to a ReportLab Image object."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    width  = width_cm * cm
    aspect = fig.get_figheight() / fig.get_figwidth()
    return Image(buf, width=width, height=width * aspect)


def _build_global_chart(summary: dict, feature_names: list) -> plt.Figure:
    top_idx      = summary["top_idx"]
    top_features = [feature_names[i] for i in top_idx]
    top_sign     = [float(summary["mean_sign"][i]) for i in top_idx]
    top_abs      = [float(summary["mean_abs"][i])  for i in top_idx]
    total        = summary["total_abs"]
    bar_vals     = [a if s >= 0 else -a for a, s in zip(top_abs, top_sign)]
    bar_colors   = ["#28a745" if s >= 0 else "#dc3545" for s in top_sign]
    max_abs      = max(top_abs) if top_abs else 1

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(top_features)), bar_vals, color=bar_colors, height=0.6)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.axvline(0, color="black", linewidth=1.2)

    for i, (bv, av) in enumerate(zip(bar_vals, top_abs)):
        pct    = (av / total * 100) if total > 0 else 0
        offset = max_abs * 0.02
        if bv >= 0:
            ax.text(bv + offset, i, f"{pct:.1f}%", va="center", ha="left",  fontsize=8, color="#333")
        else:
            ax.text(bv - offset, i, f"{pct:.1f}%", va="center", ha="right", fontsize=8, color="#333")

    ax.set_xlabel("← Decreases prediction  |  Increases prediction →", fontsize=10)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    legend_elements = [
        Patch(facecolor="#28a745", label="Increases prediction"),
        Patch(facecolor="#dc3545", label="Decreases prediction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def _build_local_chart(local_shap: np.ndarray, feature_names: list) -> plt.Figure:
    local_shap = np.array(local_shap).flatten()
    top_n      = min(10, len(feature_names))
    abs_shap   = np.abs(local_shap)
    top_idx    = np.argsort(abs_shap)[-top_n:]
    top_feats  = [feature_names[i] for i in top_idx]
    top_vals   = np.array([local_shap[i] for i in top_idx])
    top_abs    = np.abs(top_vals)
    total_abs  = top_abs.sum()
    bar_colors = ["#28a745" if v >= 0 else "#dc3545" for v in top_vals]
    max_abs    = top_abs.max() if len(top_abs) > 0 else 1

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(top_idx)), top_vals, color=bar_colors, height=0.6)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_feats, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)

    for i, (val, av) in enumerate(zip(top_vals, top_abs)):
        pct    = (av / total_abs * 100) if total_abs > 0 else 0
        offset = max_abs * 0.02
        if val >= 0:
            ax.text(val + offset, i, f"{pct:.1f}%", va="center", ha="left",  fontsize=8, color="#333")
        else:
            ax.text(val - offset, i, f"{pct:.1f}%", va="center", ha="right", fontsize=8, color="#333")

    ax.set_xlabel("← Decreases prediction  |  Increases prediction →", fontsize=10)
    ax.set_title("Local Feature Contributions (This Prediction)", fontsize=13, fontweight="bold")
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    plt.tight_layout()
    return fig


# ── Main report builder ───────────────────────────────────────────────────────

def generate_pdf_report(
    model_name:       str,
    task:             str,
    target_name:      str,
    metrics:          dict,
    feature_names:    list,
    global_summary:   dict,
    dataset_info:     dict,
    is_imbalanced:    bool        = False,
    imbalance_strategy: str       = "None",
    dropped_columns:  list        = None,
    local_shap:       np.ndarray  = None,
    local_row_data:   dict        = None,
    local_prediction: str         = None,
    username:         str         = "Unknown",
) -> bytes:
    """
    Generate a complete PDF report and return as bytes.
    Caller writes bytes to disk or serves as download.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=24, textColor=PRIMARY, spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, textColor=DARK, spaceAfter=6, spaceBefore=16,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, textColor=PRIMARY, spaceAfter=4, spaceBefore=12,
    )
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=6, leading=15,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, textColor=GREY, spaceAfter=4, leading=13,
    )
    small_bold = ParagraphStyle(
        "SmallBold", parent=styles["Normal"],
        fontSize=10, textColor=DARK, spaceAfter=4, fontName="Helvetica-Bold",
    )

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("SHAP Explainability Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 0.4*cm))

    cover_data = [
        ["Model Name",    model_name],
        ["Task Type",     task],
        ["Target Column", target_name],
        ["Generated By",  username],
        ["Generated At",  datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Dataset Rows",  f"{dataset_info.get('n_rows', 'N/A'):,}"],
        ["Features Used", str(dataset_info.get('n_features', 'N/A'))],
    ]
    cover_table = Table(cover_data, colWidths=[5*cm, 11*cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("TEXTCOLOR",  (0, 0), (0, -1), DARK),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
        ("BOX",        (0, 0), (-1, -1), 0.5, GREY),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, GREY),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.5*cm))

    # Data quality notes
    if dropped_columns:
        story.append(Paragraph("⚠ Data Quality Notes", h2))
        for col, reason in (dropped_columns or []):
            story.append(Paragraph(f"• Column <b>{col}</b> was automatically removed: {reason}", body))

    if is_imbalanced:
        story.append(Paragraph(
            f"⚠ Imbalanced dataset detected. Strategy applied: <b>{imbalance_strategy}</b>. "
            "This ensures the model does not simply predict the majority class every time.",
            body,
        ))

    story.append(PageBreak())

    # ── Model Performance ─────────────────────────────────────────────────────
    story.append(Paragraph("Model Performance", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY))
    story.append(Spacer(1, 0.3*cm))

    if task == "Classification":
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("f1",  0)
        auc = metrics.get("roc_auc", "N/A")

        perf_data = [
            ["Metric",    "Value",              "What This Means"],
            ["Accuracy",  f"{acc:.2%}",         _interpret_accuracy(acc, is_imbalanced)],
            ["F1 Score",  f"{f1:.3f}",          _interpret_f1(f1)],
            ["ROC-AUC",   str(auc) if auc == "N/A" else f"{float(auc):.3f}", _interpret_auc(auc)],
        ]
    else:
        r2   = metrics.get("r2",   0)
        mae  = metrics.get("mae",  0)
        rmse = metrics.get("rmse", 0)

        perf_data = [
            ["Metric", "Value",         "What This Means"],
            ["R²",     f"{r2:.3f}",     _interpret_r2(r2)],
            ["MAE",    f"{mae:.3f}",    f"On average, predictions are off by {mae:.3f} units."],
            ["RMSE",   f"{rmse:.3f}",   f"Root mean squared error: {rmse:.3f}."],
        ]

    perf_table = Table(perf_data, colWidths=[3.5*cm, 2.5*cm, 10*cm])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("BOX",        (0, 0), (-1, -1), 0.5, GREY),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, GREY),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("PADDING",    (0, 0), (-1, -1), 7),
    ]))
    story.append(perf_table)
    story.append(PageBreak())

    # ── Global Explainability ─────────────────────────────────────────────────
    story.append(Paragraph("Global Explainability — What Drives This Model?", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "The chart below shows which features have the most influence on the model's predictions "
        "across the entire dataset. Green bars (pointing right) mean the feature on average "
        "increases the predicted value. Red bars (pointing left) mean it decreases it. "
        "The percentage shows how much of the total influence that feature contributes.",
        body,
    ))
    story.append(Spacer(1, 0.3*cm))

    global_fig = _build_global_chart(global_summary, feature_names)
    story.append(_fig_to_image(global_fig, width_cm=16))
    story.append(Spacer(1, 0.4*cm))

    # Plain English feature explanations
    story.append(Paragraph("Feature-by-Feature Explanation", h2))
    top_idx   = global_summary["top_idx"]
    mean_abs  = global_summary["mean_abs"]
    mean_sign = global_summary["mean_sign"]
    total     = global_summary["total_abs"]

    for rank, idx in enumerate(top_idx[:6]):
        idx  = int(idx)
        feat = feature_names[idx]
        pct  = (float(mean_abs[idx]) / total * 100) if total > 0 else 0
        sv   = float(mean_sign[idx])
        direction = "increases" if sv >= 0 else "decreases"
        color_hex = "#28a745" if sv >= 0 else "#dc3545"

        story.append(Paragraph(
            f'<b><font color="{color_hex}">#{rank+1} — {feat}</font></b> '
            f'({pct:.1f}% of total influence)',
            small_bold,
        ))
        story.append(Paragraph(
            f"On average across all predictions, <b>{feat}</b> <b>{direction}</b> the model's output. "
            f"It accounts for <b>{pct:.1f}%</b> of the model's total decision-making influence.",
            body,
        ))

    story.append(PageBreak())

    # ── Local Explainability (optional) ──────────────────────────────────────
    if local_shap is not None:
        story.append(Paragraph("Local Explainability — Why This Specific Prediction?", h1))
        story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY))
        story.append(Spacer(1, 0.2*cm))

        if local_prediction is not None:
            story.append(Paragraph(
                f"Model Prediction: <b>{local_prediction}</b>",
                ParagraphStyle("pred", parent=body, fontSize=13, textColor=PRIMARY),
            ))
            story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph(
            "The chart below explains why the model made this specific prediction. "
            "Each bar shows how much a feature pushed the prediction higher (green, right) "
            "or lower (red, left) for this individual case.",
            body,
        ))
        story.append(Spacer(1, 0.3*cm))

        local_fig = _build_local_chart(local_shap, feature_names)
        story.append(_fig_to_image(local_fig, width_cm=16))
        story.append(Spacer(1, 0.4*cm))

        # Input values table
        if local_row_data:
            story.append(Paragraph("Input Values for This Prediction", h2))
            row_table_data = [["Feature", "Value"]] + [
                [k, str(round(v, 4)) if isinstance(v, float) else str(v)]
                for k, v in local_row_data.items()
            ]
            row_table = Table(row_table_data, colWidths=[8*cm, 8*cm])
            row_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("BOX",        (0, 0), (-1, -1), 0.5, GREY),
                ("INNERGRID",  (0, 0), (-1, -1), 0.25, GREY),
                ("PADDING",    (0, 0), (-1, -1), 6),
            ]))
            story.append(row_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY))
    story.append(Paragraph(
        f"Generated by SHAP Explainability Tool · {datetime.now().strftime('%Y-%m-%d %H:%M')} · User: {username}",
        ParagraphStyle("footer", parent=caption, alignment=TA_CENTER),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
