"""
backend/fairness.py — Bias and Fairness Detection.

Regulated industries (banking, healthcare, HR, insurance) require proof
that models don't discriminate based on protected attributes.

EU AI Act, GDPR, US Equal Credit Opportunity Act, and HIPAA all have
requirements around algorithmic fairness. This module:

1. Detects if protected-sounding columns appear in top SHAP features
2. Computes demographic parity: are prediction rates equal across groups?
3. Computes equal opportunity: are true positive rates equal across groups?
4. Flags disparate impact (the 80% rule used in US employment law)
5. Produces plain-English fairness report

Protected attribute detection uses keyword matching — not perfect,
but catches the most common cases (age, gender, race, religion, etc.)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Protected attribute keywords ─────────────────────────────────────────────

PROTECTED_KEYWORDS = [
    "age", "gender", "sex", "race", "ethnicity", "nationality", "religion",
    "disability", "marital", "pregnant", "pregnancy", "color", "colour",
    "origin", "tribe", "caste", "sexual", "orientation", "birth",
    "citizen", "immigrant", "veteran", "height", "weight",
]

# The 80% rule: if selection rate of protected group < 80% of majority, flag it
DISPARATE_IMPACT_THRESHOLD = 0.80


def detect_protected_columns(
    column_names: list,
    shap_top_features: list,
) -> dict:
    """
    Check if any protected-sounding columns are in the dataset or top SHAP features.

    Returns dict with:
        all_protected     — all column names that match protected keywords
        top_shap_protected — protected columns that appear in top SHAP features
        has_risk          — bool: True if protected cols are in top SHAP
        warning           — plain-English warning message
    """
    all_protected = [
        col for col in column_names
        if any(kw in col.lower() for kw in PROTECTED_KEYWORDS)
    ]

    top_shap_protected = [
        feat for feat in shap_top_features
        if any(kw in feat.lower() for kw in PROTECTED_KEYWORDS)
    ]

    has_risk = len(top_shap_protected) > 0

    warning = ""
    if has_risk:
        warning = (
            f"⚠️ <b>Potential Fairness Risk:</b> The following protected attributes "
            f"appear in the model's top influential features: "
            f"<b>{', '.join(top_shap_protected)}</b>. "
            f"This may indicate the model is making decisions partly based on "
            f"protected characteristics. Review carefully before deployment "
            f"in regulated contexts."
        )
    elif all_protected:
        warning = (
            f"ℹ️ Protected attributes detected in dataset: "
            f"<b>{', '.join(all_protected)}</b>. "
            f"These do not appear in top SHAP features, which is a good sign. "
            f"Still verify fairness metrics below."
        )

    return {
        "all_protected":      all_protected,
        "top_shap_protected": top_shap_protected,
        "has_risk":           has_risk,
        "warning":            warning,
    }


def compute_group_fairness(
    df:               pd.DataFrame,
    predictions:      np.ndarray,
    protected_col:    str,
    actual_labels:    Optional[pd.Series] = None,
    task:             str = "Classification",
) -> dict:
    """
    Compute fairness metrics for a binary protected attribute column.

    Metrics:
        demographic_parity  — equal prediction rates across groups
        disparate_impact    — 80% rule (minority rate / majority rate)
        equal_opportunity   — equal true positive rates (if actuals provided)

    Returns dict with per-group stats and fairness flags.
    """
    if task != "Classification":
        return {"error": "Fairness metrics only available for classification tasks."}

    if protected_col not in df.columns:
        return {"error": f"Column '{protected_col}' not found in dataset."}

    preds = np.array(predictions)
    groups = df[protected_col].dropna().unique()

    if len(groups) > 10:
        return {
            "error": f"'{protected_col}' has {len(groups)} unique values — too many groups for fairness analysis. Use a binary or low-cardinality column."
        }

    group_stats = []
    for group in sorted(groups, key=str):
        mask = df[protected_col] == group
        group_preds = preds[mask.values[:len(preds)]]

        if len(group_preds) == 0:
            continue

        pred_rate = float(np.mean(group_preds == 1)) if len(group_preds) > 0 else 0

        stat = {
            "group":     str(group),
            "n":         int(mask.sum()),
            "pred_rate": round(pred_rate, 4),
            "pred_pct":  f"{pred_rate:.1%}",
        }

        # Equal opportunity (TPR) if actuals provided
        if actual_labels is not None:
            actual = actual_labels.values[:len(preds)]
            group_actual = actual[mask.values[:len(preds)]]
            tp = int(np.sum((group_preds == 1) & (group_actual == 1)))
            fn = int(np.sum((group_preds == 0) & (group_actual == 1)))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else None
            stat["tpr"] = round(tpr, 4) if tpr is not None else None
            stat["tpr_pct"] = f"{tpr:.1%}" if tpr is not None else "N/A"

        group_stats.append(stat)

    if len(group_stats) < 2:
        return {"error": "Need at least 2 groups for fairness comparison."}

    # Demographic parity
    pred_rates  = [s["pred_rate"] for s in group_stats]
    max_rate    = max(pred_rates)
    min_rate    = min(pred_rates)
    max_group   = group_stats[pred_rates.index(max_rate)]["group"]
    min_group   = group_stats[pred_rates.index(min_rate)]["group"]
    disparity   = max_rate - min_rate

    # Disparate impact (80% rule)
    disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0
    fails_80_rule    = disparate_impact < DISPARATE_IMPACT_THRESHOLD

    # Equal opportunity gap
    eo_gap = None
    if actual_labels is not None and all(s.get("tpr") is not None for s in group_stats):
        tprs   = [s["tpr"] for s in group_stats]
        eo_gap = max(tprs) - min(tprs)

    # Plain English verdict
    if fails_80_rule:
        verdict = (
            f"🚨 <b>Disparate Impact Detected</b> — The 80% rule is violated. "
            f"Group <b>{max_group}</b> has a {max_rate:.1%} positive prediction rate "
            f"while group <b>{min_group}</b> has only {min_rate:.1%} "
            f"(ratio: {disparate_impact:.2f}, threshold: 0.80). "
            f"This model may be discriminatory under US employment law and EU AI Act guidelines."
        )
        verdict_color = "#dc3545"
    elif disparity > 0.10:
        verdict = (
            f"⚠️ <b>Moderate Disparity</b> — Prediction rates differ by {disparity:.1%} "
            f"between groups. Passes the 80% rule but warrants review."
        )
        verdict_color = "#f0a500"
    else:
        verdict = (
            f"✅ <b>Fair</b> — Prediction rates are similar across groups "
            f"(max disparity: {disparity:.1%}). No significant bias detected."
        )
        verdict_color = "#28a745"

    return {
        "group_stats":       group_stats,
        "disparity":         round(disparity, 4),
        "disparate_impact":  round(disparate_impact, 4),
        "fails_80_rule":     fails_80_rule,
        "eo_gap":            round(eo_gap, 4) if eo_gap is not None else None,
        "verdict":           verdict,
        "verdict_color":     verdict_color,
        "max_group":         max_group,
        "min_group":         min_group,
    }


def full_fairness_report(
    df:            pd.DataFrame,
    predictions:   np.ndarray,
    column_names:  list,
    shap_top_features: list,
    actual_labels: Optional[pd.Series] = None,
    task:          str = "Classification",
) -> dict:
    """
    Run full fairness analysis: detection + metrics for all protected columns found.
    Returns dict ready for UI rendering.
    """
    detection = detect_protected_columns(column_names, shap_top_features)

    group_results = {}
    for col in detection["all_protected"]:
        if col in df.columns:
            result = compute_group_fairness(
                df, predictions, col, actual_labels, task
            )
            if "error" not in result:
                group_results[col] = result

    return {
        "detection":     detection,
        "group_results": group_results,
        "has_protected": len(detection["all_protected"]) > 0,
        "has_risk":      detection["has_risk"],
    }
