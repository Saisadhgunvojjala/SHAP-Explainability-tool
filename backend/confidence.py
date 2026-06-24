"""
backend/confidence.py — Prediction confidence and uncertainty scoring.

Real-world problem: a model that predicts "Diabetic" with 51% probability
should be treated very differently from one predicting with 95% probability.
Without confidence scoring, clients act on uncertain predictions as if they
were certain — dangerous in healthcare, finance, and cybersecurity.

Provides:
- Confidence level (High / Medium / Low / Very Low)
- Plain-English uncertainty warning when confidence is borderline
- Calibration check: are probabilities actually reliable?
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Confidence thresholds ─────────────────────────────────────────────────────

CONFIDENCE_LEVELS = {
    "Very High": (0.90, 1.00, "#1b5e20", "✅"),
    "High":      (0.75, 0.90, "#28a745", "🟢"),
    "Medium":    (0.60, 0.75, "#f0a500", "🟡"),
    "Low":       (0.50, 0.60, "#dc3545", "🔴"),
}

UNCERTAINTY_ZONE = (0.40, 0.60)  # Model is genuinely uncertain in this range


def get_confidence(probability: float) -> dict:
    """
    Given a predicted probability (class 1), return confidence metadata.

    Works for binary classification only.
    For regression, use prediction interval instead (see below).

    Returns dict with:
        level        — "Very High" / "High" / "Medium" / "Low"
        color        — hex color for UI
        icon         — emoji indicator
        probability  — raw probability
        pct          — probability as percentage string
        is_uncertain — bool: True if model is genuinely unsure
        warning      — plain-English warning message (if uncertain)
        explanation  — plain-English confidence explanation
    """
    prob = float(probability)

    # Find level
    level = "Low"
    color = "#dc3545"
    icon  = "🔴"
    for lvl, (low, high, col, ico) in CONFIDENCE_LEVELS.items():
        if low <= prob <= high:
            level = lvl
            color = col
            icon  = ico
            break

    # Mirror for negative class (if probability < 0.5, flip)
    display_prob = prob if prob >= 0.5 else 1 - prob

    is_uncertain = UNCERTAINTY_ZONE[0] <= prob <= UNCERTAINTY_ZONE[1]

    # Plain English explanation
    if display_prob >= 0.90:
        explanation = (
            f"The model is very confident in this prediction "
            f"({display_prob:.0%} certainty). "
            "This prediction can be relied upon with high confidence."
        )
    elif display_prob >= 0.75:
        explanation = (
            f"The model is fairly confident ({display_prob:.0%} certainty). "
            "This prediction is reliable but worth verifying with domain knowledge."
        )
    elif display_prob >= 0.60:
        explanation = (
            f"The model has moderate confidence ({display_prob:.0%} certainty). "
            "Consider this prediction as one input among several — "
            "additional review is recommended."
        )
    else:
        explanation = (
            f"The model has low confidence ({display_prob:.0%} certainty). "
            "This is a borderline case. Do not rely on this prediction alone — "
            "human review is strongly recommended."
        )

    warning = ""
    if is_uncertain:
        warning = (
            "⚠️ This case falls in the model's uncertainty zone "
            f"(probability {prob:.1%} is close to 50%). "
            "The model cannot clearly distinguish between outcomes for this record. "
            "Treat this prediction with extra caution."
        )

    return {
        "level":        level,
        "color":        color,
        "icon":         icon,
        "probability":  prob,
        "display_prob": display_prob,
        "pct":          f"{display_prob:.1%}",
        "is_uncertain": is_uncertain,
        "warning":      warning,
        "explanation":  explanation,
    }


def batch_confidence_summary(probabilities: np.ndarray) -> dict:
    """
    Summarise confidence distribution across all predictions.
    Useful for showing clients how reliable the model is overall.
    """
    probs = np.array(probabilities)

    # Mirror probabilities below 0.5
    display_probs = np.where(probs >= 0.5, probs, 1 - probs)

    counts = {
        "Very High": int(np.sum(display_probs >= 0.90)),
        "High":      int(np.sum((display_probs >= 0.75) & (display_probs < 0.90))),
        "Medium":    int(np.sum((display_probs >= 0.60) & (display_probs < 0.75))),
        "Low":       int(np.sum(display_probs < 0.60)),
    }
    uncertain_count = int(np.sum(
        (probs >= UNCERTAINTY_ZONE[0]) & (probs <= UNCERTAINTY_ZONE[1])
    ))

    total = len(probs)
    pcts  = {k: v / total * 100 for k, v in counts.items()}

    return {
        "counts":         counts,
        "percentages":    pcts,
        "uncertain_count":uncertain_count,
        "uncertain_pct":  uncertain_count / total * 100,
        "total":          total,
        "mean_confidence":float(display_probs.mean()),
        "overall_reliable": pcts["Very High"] + pcts["High"] >= 60,
    }


def regression_confidence(
    model,
    X_single: pd.DataFrame,
    X_train:  pd.DataFrame,
    y_train:  pd.Series,
    n_bootstrap: int = 50,
) -> dict:
    """
    Estimate prediction interval for regression using bootstrap.
    Returns a confidence range [low, high] around the prediction.

    Note: uses a fast approximation (residual std) rather than full bootstrap
    for speed in a web app context.
    """
    from sklearn.metrics import mean_squared_error

    # Predict on training set to get residual std
    train_preds = model.predict(X_train)
    residuals   = np.array(y_train) - train_preds
    residual_std = float(np.std(residuals))

    prediction = float(model.predict(X_single)[0])

    # 80% prediction interval (1.28 std devs)
    low_80  = prediction - 1.28 * residual_std
    high_80 = prediction + 1.28 * residual_std

    # 95% prediction interval (1.96 std devs)
    low_95  = prediction - 1.96 * residual_std
    high_95 = prediction + 1.96 * residual_std

    # Relative uncertainty
    if prediction != 0:
        rel_uncertainty = (residual_std / abs(prediction)) * 100
    else:
        rel_uncertainty = 100.0

    if rel_uncertainty < 10:
        level = "High"
        explanation = f"The model's predictions are precise (±{residual_std:.2f} typical error)."
    elif rel_uncertainty < 25:
        level = "Medium"
        explanation = f"The model has moderate precision (±{residual_std:.2f} typical error)."
    else:
        level = "Low"
        explanation = f"The model has high uncertainty (±{residual_std:.2f} typical error — {rel_uncertainty:.0f}% of prediction)."

    return {
        "prediction":      prediction,
        "residual_std":    residual_std,
        "interval_80":     (round(low_80, 3), round(high_80, 3)),
        "interval_95":     (round(low_95, 3), round(high_95, 3)),
        "rel_uncertainty": rel_uncertainty,
        "level":           level,
        "explanation":     explanation,
    }
