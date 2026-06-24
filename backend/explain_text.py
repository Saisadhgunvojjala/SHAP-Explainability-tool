"""
backend/explain_text.py — Plain-English SHAP explanation generator.

Produces two levels of explanation for every feature:
  - Non-technical: plain language a business user or patient can read
  - Technical:     includes SHAP values, direction, percentile context

Design principles:
  1. Never say "SHAP value" to a non-technical user
  2. Always anchor to the actual feature value ("Your glucose was 180")
  3. Use relative language ("much higher than average") not raw numbers
  4. Explain what the prediction IS before explaining why
  5. Keep sentences under 20 words
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Magnitude descriptors ─────────────────────────────────────────────────────

def _magnitude(pct: float) -> str:
    """Convert influence % to plain-English magnitude."""
    if pct >= 30:
        return "the strongest factor"
    elif pct >= 20:
        return "a very strong factor"
    elif pct >= 12:
        return "an important factor"
    elif pct >= 6:
        return "a moderate factor"
    else:
        return "a minor factor"


def _relative_value(val, col_mean: float, col_std: float) -> str:
    """
    Describe a value relative to the dataset average.
    e.g. "much higher than average", "slightly below average"
    """
    if col_std == 0:
        return "at the typical level"
    z = (val - col_mean) / col_std
    if z > 2:
        return "much higher than average"
    elif z > 0.75:
        return "above average"
    elif z > -0.75:
        return "around average"
    elif z > -2:
        return "below average"
    else:
        return "much lower than average"


def _format_value(val) -> str:
    """Format a value for display — round floats, keep strings."""
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return f"{val:.2f}"
    return str(val)


# ── Global explanation text ───────────────────────────────────────────────────

def global_explanation_text(
    feature_names:  list,
    mean_abs:       np.ndarray,
    mean_sign:      np.ndarray,
    X_test:         pd.DataFrame,
    task:           str,
    target_name:    str,
    top_idx:        list,
    total:          float,
    mode:           str = "both",  # "simple", "technical", "both"
) -> list[dict]:
    """
    Generate rich explanation dicts for each top feature (global).

    Returns list of dicts with keys:
        feature, value_context, pct, sv, color, icon,
        simple_text, technical_text, magnitude
    """
    explanations = []

    for rank, idx in enumerate(top_idx[:8]):
        idx   = int(idx)
        feat  = feature_names[idx]
        pct   = (float(mean_abs[idx]) / total * 100) if total > 0 else 0
        sv    = float(mean_sign[idx])
        mag   = _magnitude(pct)
        color = "#28a745" if sv >= 0 else "#dc3545"
        icon  = "📈" if sv >= 0 else "📉"

        # Average value in test set
        avg_val = float(X_test.iloc[:, idx].mean()) if idx < X_test.shape[1] else None

        # Direction phrasing
        if sv >= 0:
            direction_simple = "tends to increase"
            direction_tech   = "positively influences"
        else:
            direction_simple = "tends to decrease"
            direction_tech   = "negatively influences"

        # Simple explanation (non-technical)
        if avg_val is not None:
            simple = (
                f"<b>{feat}</b> is {mag} in this model. "
                f"On average, when <b>{feat}</b> is higher, "
                f"it {direction_simple} the predicted {target_name}. "
                f"The typical value in your dataset is <b>{_format_value(avg_val)}</b>."
            )
        else:
            simple = (
                f"<b>{feat}</b> is {mag} in this model — "
                f"it {direction_simple} the predicted {target_name}."
            )

        # Technical explanation
        technical = (
            f"Mean |SHAP|: <code>{abs(sv):.4f}</code> &nbsp;·&nbsp; "
            f"Accounts for <b>{pct:.1f}%</b> of total model influence. "
            f"Average signed SHAP: <code>{sv:+.4f}</code> — "
            f"this feature {direction_tech} the model output on average."
        )

        # What this means in context
        if pct >= 20:
            context = f"⭐ This is one of the TOP drivers of the model's decisions."
        elif pct >= 10:
            context = f"This feature plays a significant role in most predictions."
        else:
            context = f"This feature has a smaller but still measurable effect."

        explanations.append({
            "rank":          rank + 1,
            "feature":       feat,
            "pct":           pct,
            "sv":            sv,
            "color":         color,
            "icon":          icon,
            "magnitude":     mag,
            "avg_val":       avg_val,
            "simple_text":   simple,
            "technical_text":technical,
            "context":       context,
        })

    return explanations


# ── Local explanation text ────────────────────────────────────────────────────

def local_explanation_text(
    feature_names:  list,
    local_shap:     np.ndarray,
    raw_row:        dict,
    prediction,
    task:           str,
    target_name:    str,
    X_train:        Optional[pd.DataFrame] = None,
    positive_class: str = None,
    negative_class: str = None,
) -> tuple[str, list[dict]]:
    """
    Generate a prediction summary + per-feature explanations for one row.

    Returns:
        summary_text  — one paragraph explaining the prediction overall
        explanations  — list of feature dicts with simple + technical text
    """
    local_arr   = np.array(local_shap).flatten()
    abs_arr     = np.abs(local_arr)
    total_abs   = abs_arr.sum()
    top_idx     = np.argsort(abs_arr)[::-1][:6]

    # ── Prediction summary ────────────────────────────────────────────────────
    pred_str = str(prediction) if task == "Classification" else f"{float(prediction):.2f}"

    if task == "Classification":
        # Count how many features push toward vs away from prediction
        n_pushing_up  = sum(1 for i in top_idx if local_arr[int(i)] > 0)
        n_pushing_down= sum(1 for i in top_idx if local_arr[int(i)] < 0)
        top_feat      = feature_names[int(top_idx[0])]
        top_val       = raw_row.get(top_feat, "N/A")

        if n_pushing_up > n_pushing_down:
            summary = (
                f"The model predicted <b>{pred_str}</b> for this case. "
                f"Most features in this record pushed the prediction in this direction. "
                f"The single biggest reason was <b>{top_feat}</b> "
                f"(value: <b>{_format_value(top_val)}</b>), "
                f"which had the strongest influence on this outcome."
            )
        else:
            summary = (
                f"The model predicted <b>{pred_str}</b> for this case. "
                f"Despite some features pushing in the opposite direction, "
                f"<b>{top_feat}</b> (value: <b>{_format_value(top_val)}</b>) "
                f"was the dominant factor driving this prediction."
            )
    else:
        top_feat = feature_names[int(top_idx[0])]
        top_val  = raw_row.get(top_feat, "N/A")
        summary  = (
            f"The model predicted <b>{pred_str}</b> for {target_name}. "
            f"The most influential factor was <b>{top_feat}</b> "
            f"(value: <b>{_format_value(top_val)}</b>). "
            f"See below for a feature-by-feature breakdown."
        )

    # ── Per-feature explanations ──────────────────────────────────────────────
    explanations = []

    for idx in top_idx:
        idx   = int(idx)
        feat  = feature_names[idx]
        sv    = local_arr[idx]
        pct   = (abs(sv) / total_abs * 100) if total_abs > 0 else 0
        mag   = _magnitude(pct)
        color = "#28a745" if sv >= 0 else "#dc3545"
        icon  = "📈" if sv >= 0 else "📉"

        raw_val   = raw_row.get(feat, None)
        val_str   = _format_value(raw_val) if raw_val is not None else "N/A"

        # Relative context using training data stats
        rel_context = ""
        if X_train is not None and feat in X_train.columns:
            col_data = X_train[feat]
            if pd.api.types.is_numeric_dtype(col_data):
                try:
                    numeric_val = float(raw_val) if raw_val is not None else col_data.mean()
                    rel_context = _relative_value(
                        numeric_val,
                        float(col_data.mean()),
                        float(col_data.std()),
                    )
                except (ValueError, TypeError):
                    rel_context = ""  # raw_val is categorical string, skip relative context

        # Direction phrasing — use actual class names so green=approval is unambiguous
        _pos = positive_class or target_name
        _neg = negative_class or target_name
        if task == "Classification":
            if sv >= 0:
                effect_simple = f"increases the chance of <b>{_pos}</b>"
                effect_tech   = f"positive contribution → pushes toward {_pos}"
            else:
                effect_simple = f"decreases the chance of <b>{_pos}</b> (pushes toward {_neg})"
                effect_tech   = f"negative contribution → pushes toward {_neg}"
        else:
            if sv >= 0:
                effect_simple = f"increased the predicted {target_name}"
                effect_tech   = "positive contribution to prediction"
            else:
                effect_simple = f"decreased the predicted {target_name}"
                effect_tech   = "negative contribution to prediction"

        # Simple explanation
        if rel_context:
            simple = (
                f"<b>{feat}</b> was <b>{val_str}</b>, which is {rel_context}. "
                f"This {effect_simple} — it is {mag} for this prediction."
            )
        else:
            simple = (
                f"<b>{feat}</b> = <b>{val_str}</b>. "
                f"This {effect_simple} — it is {mag} for this prediction."
            )

        # Technical explanation
        technical = (
            f"SHAP value: <code>{sv:+.4f}</code> &nbsp;·&nbsp; "
            f"Contribution: <b>{pct:.1f}%</b> of this row's total impact &nbsp;·&nbsp; "
            f"{effect_tech}."
        )

        explanations.append({
            "rank":          len(explanations) + 1,
            "feature":       feat,
            "value":         val_str,
            "pct":           pct,
            "sv":            sv,
            "color":         color,
            "icon":          icon,
            "magnitude":     mag,
            "rel_context":   rel_context,
            "simple_text":   simple,
            "technical_text":technical,
            "effect_simple": effect_simple,
        })

    return summary, explanations


# ── Render helpers ────────────────────────────────────────────────────────────

def render_explanation_card(exp: dict, show_technical: bool = True) -> str:
    """
    Render one explanation dict as an HTML card string.
    Pass to st.markdown(..., unsafe_allow_html=True).
    """
    color   = exp["color"]
    icon    = exp["icon"]
    rank    = exp.get("rank", "")
    feature = exp["feature"]
    pct     = exp["pct"]
    simple  = exp["simple_text"]
    tech    = exp.get("technical_text", "")
    context = exp.get("context", "")

    tech_block = (
        f'<div style="margin-top:8px;padding:8px;background:#f8f9fa;'
        f'border-radius:4px;font-size:0.85em;color:#555;">'
        f'🔬 <b>Technical:</b> {tech}</div>'
    ) if show_technical and tech else ""

    context_block = (
        f'<div style="margin-top:6px;font-size:0.85em;color:#888;">{context}</div>'
    ) if context else ""

    return f"""
<div class="explanation-box" style="border-left-color:{color};">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span><b>{icon} #{rank} — {feature}</b></span>
    <span style="background:{color};color:white;padding:2px 10px;
          border-radius:12px;font-size:0.85em;font-weight:bold;">{pct:.1f}% influence</span>
  </div>
  <div style="margin-top:8px;font-size:0.95em;">{simple}</div>
  {tech_block}
  {context_block}
</div>"""


def render_summary_box(summary_text: str, prediction, task: str, target_name: str,
                       label_encoder=None) -> str:
    """Render the prediction summary box with decoded class label if available."""
    if task == "Classification" and label_encoder is not None:
        try:
            pred_str = str(label_encoder.inverse_transform([int(prediction)])[0])
        except Exception:
            pred_str = str(prediction)
    elif task == "Classification":
        pred_str = str(prediction)
    else:
        pred_str = f"{float(prediction):.2f}"
    return f"""
<div style="background:#f0f4ff;padding:20px;border-radius:12px;
     border:2px solid #667eea;margin:16px 0;">
  <h3 style="margin:0 0 10px 0;color:#667eea;">🎯 Prediction: {pred_str}</h3>
  <p style="margin:0;font-size:0.95em;line-height:1.6;">{summary_text}</p>
</div>"""