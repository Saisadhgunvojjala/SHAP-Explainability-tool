"""
backend/whatif.py — What-If and Counterfactual Analysis.

Works on ANY dataset: credit, healthcare, HR, manufacturing, cybersecurity, loan.

Key fixes vs original:
1. Categorical counterfactuals — iterate over actual category values, not float linspace.
2. Numeric ranges taken from original_df (unscaled), not from encoded X_train.
3. Low-cardinality categoricals (OrdinalEncoder path) handled correctly at inference.
4. Regression counterfactuals — show sensitivity range, not just "unsupported".
5. apply_inference_preprocessing used for all transforms — single source of truth.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Shared preprocessing helper ───────────────────────────────────────────────

def _transform_row(row_dict: dict, original_columns: list, encoding_map: dict,
                   preprocessor, feature_names: list) -> pd.DataFrame:
    """
    Transform one raw row (original values, strings allowed) into model-ready DataFrame.
    Handles both freq-encoded high-card and OrdinalEncoder low-card categoricals.
    Raises ValueError with a clear message on failure.
    """
    from utils.preprocess import apply_inference_preprocessing
    df = pd.DataFrame([row_dict])
    return apply_inference_preprocessing(df, original_columns, encoding_map,
                                         preprocessor, feature_names)


# ── whatif_predict ────────────────────────────────────────────────────────────

def whatif_predict(
    model,
    explainer,
    preprocessor,
    encoding_map:     dict,
    feature_names:    list,
    original_columns: list,
    modified_values:  dict,
    task:             str,
) -> dict:
    """
    Given a modified row of original feature values (strings for categoricals OK),
    return new prediction + SHAP values.

    For multi-class, SHAP is extracted for the PREDICTED class so the chart
    always shows meaningful values (not near-zero values for the wrong class).
    """
    X_single = _transform_row(
        modified_values, original_columns, encoding_map, preprocessor, feature_names
    )

    prediction = model.predict(X_single)[0]
    pred_class = int(prediction) if task == "Classification" else None

    # All class probabilities (works for binary and multi-class)
    all_probs = None
    probability = None
    if task == "Classification":
        try:
            all_probs = model.predict_proba(X_single)[0].tolist()
            probability = all_probs[pred_class]  # confidence for predicted class
        except Exception:
            probability = None

    # SHAP — always for the PREDICTED class so bars are meaningful
    try:
        from backend.shap_explainer import compute_single_row_shap
        shap_arr = compute_single_row_shap(explainer, X_single,
                                           predicted_class=pred_class)
    except Exception:
        try:
            raw = explainer.shap_values(X_single)
            if isinstance(raw, list):
                idx = pred_class if pred_class is not None and pred_class < len(raw) else 0
                shap_arr = np.array(raw[idx]).flatten()
            else:
                shap_arr = np.array(raw).flatten()
        except Exception:
            shap_arr = np.zeros(len(feature_names))

    return {
        "prediction":    prediction,
        "pred_class":    pred_class,
        "probability":   probability,
        "all_probs":     all_probs,
        "shap_values":   shap_arr.tolist(),
        "feature_names": feature_names,
        "X_processed":   X_single,
    }


# ── find_counterfactual ───────────────────────────────────────────────────────

def find_counterfactual(
    model,
    preprocessor,
    encoding_map:     dict,
    feature_names:    list,
    original_columns: list,
    original_row:     dict,
    X_train:          pd.DataFrame,
    task:             str,
    original_df:      pd.DataFrame = None,
    explainer = None,              # needed for SHAP-based feature ranking
    n_steps:          int = 25,
    top_n_features:   int = 5,
) -> list[dict]:
    """
    Find minimum changes to flip the prediction (classification) or
    show sensitivity range (regression).

    Works on ANY dataset — handles:
    - Numeric continuous features (linspace over original unscaled range)
    - Low-cardinality categoricals (iterate all possible category values)
    - High-cardinality freq-encoded features (iterate top-10 freq values)

    Returns list of suggestion dicts.
    """
    # ── Get original prediction ───────────────────────────────────────────────
    try:
        X_orig    = _transform_row(original_row, original_columns, encoding_map,
                                    preprocessor, feature_names)
        orig_pred = model.predict(X_orig)[0]
    except Exception as e:
        raise ValueError(f"Could not transform original row: {e}")

    is_classification = (task == "Classification")
    suggestions       = []

    # ── Build feature profiles from original_df (unscaled) ───────────────────
    # original_df has real values; X_train has scaled/encoded — don't use X_train for ranges
    ref_df = original_df if original_df is not None else pd.DataFrame()

    def _get_col_profile(col: str) -> dict:
        """Return dtype, unique values, min, max from original data."""
        if col not in ref_df.columns:
            # Fall back to original_row itself
            val = original_row.get(col)
            return {"dtype": "unknown", "uniques": [val], "min": None, "max": None}

        series = ref_df[col].dropna()
        is_num = pd.api.types.is_numeric_dtype(series)
        return {
            "dtype":   "numeric" if is_num else "categorical",
            "uniques": sorted(series.unique().tolist(), key=str),
            "min":     float(series.min()) if is_num else None,
            "max":     float(series.max()) if is_num else None,
        }

    # ── Rank features by SHAP importance to prioritise which to flip ──────────
    # Rank features by SHAP importance for the predicted class
    cols_to_try = [c for c in original_columns if c in original_row]
    try:
        from backend.shap_explainer import compute_single_row_shap
        local_shap = compute_single_row_shap(explainer, X_orig,
                                              predicted_class=int(orig_pred) if is_classification else None)
        if np.abs(local_shap).sum() > 0:
            # Sort original_columns by their SHAP importance
            shap_by_feat = {}
            for i, fn in enumerate(feature_names):
                # Map feature_name back to original_column
                for oc in original_columns:
                    from utils.preprocess import _sanitize_col
                    if _sanitize_col(oc) == fn or oc == fn:
                        shap_by_feat[oc] = abs(float(local_shap[i]))
                        break
            cols_to_try = sorted(
                [c for c in original_columns if c in original_row],
                key=lambda c: shap_by_feat.get(c, 0),
                reverse=True
            )
    except Exception:
        pass  # keep original_columns order

    # ── Try flipping each feature ─────────────────────────────────────────────
    for feat in cols_to_try:
        if len(suggestions) >= top_n_features:
            break

        profile   = _get_col_profile(feat)
        orig_val  = original_row.get(feat)

        # ── CATEGORICAL feature ───────────────────────────────────────────────
        if profile["dtype"] == "categorical" or (
            profile["dtype"] == "numeric" and len(profile["uniques"]) <= 15
            and feat in encoding_map
        ):
            candidates = [v for v in profile["uniques"] if str(v) != str(orig_val)]
            if not candidates:
                continue

            # For high-card freq-encoded: limit to top-10 by frequency
            if feat in encoding_map and len(candidates) > 10:
                freq_map = encoding_map[feat]
                candidates = sorted(candidates, key=lambda v: freq_map.get(v, 0), reverse=True)[:10]

            flip_val = None
            for cand in candidates:
                test_row = original_row.copy()
                test_row[feat] = cand
                try:
                    X_cf = _transform_row(test_row, original_columns, encoding_map,
                                          preprocessor, feature_names)
                    new_pred = model.predict(X_cf)[0]

                    if is_classification and new_pred != orig_pred:
                        flip_val = cand
                        break
                    elif not is_classification:
                        # For regression: find value that moves prediction most
                        orig_reg = float(model.predict(X_orig)[0])
                        new_reg  = float(new_pred)
                        if abs(new_reg - orig_reg) > abs(flip_val[1] - orig_reg if flip_val else 0):
                            flip_val = (cand, new_reg)
                except Exception:
                    continue

            if flip_val is not None:
                if is_classification:
                    plain = (
                        f"Change <b>{feat}</b> from "
                        f"<b>'{orig_val}'</b> → <b>'{flip_val}'</b> "
                        f"to flip the prediction."
                    )
                    suggestions.append({
                        "feature":          feat,
                        "original_value":   orig_val,
                        "suggested_value":  flip_val,
                        "change_direction": "change category",
                        "change_amount":    None,
                        "change_pct":       None,
                        "feature_type":     "categorical",
                        "plain_english":    plain,
                    })
                else:
                    cand_val, new_reg_val = flip_val
                    orig_reg_val = float(model.predict(X_orig)[0])
                    plain = (
                        f"Changing <b>{feat}</b> from <b>'{orig_val}'</b> to "
                        f"<b>'{cand_val}'</b> shifts the prediction from "
                        f"<b>{orig_reg_val:.2f}</b> to <b>{new_reg_val:.2f}</b>."
                    )
                    suggestions.append({
                        "feature":          feat,
                        "original_value":   orig_val,
                        "suggested_value":  cand_val,
                        "change_direction": "change category",
                        "change_amount":    abs(new_reg_val - orig_reg_val),
                        "change_pct":       None,
                        "feature_type":     "categorical",
                        "plain_english":    plain,
                    })

        # ── NUMERIC feature ───────────────────────────────────────────────────
        elif profile["dtype"] == "numeric":
            try:
                orig_num = float(orig_val)
            except (TypeError, ValueError):
                continue  # value is a string — skip

            feat_min = profile["min"] if profile["min"] is not None else orig_num * 0.5
            feat_max = profile["max"] if profile["max"] is not None else orig_num * 1.5

            if feat_min == feat_max:
                continue

            flip_val  = None
            direction = None
            best_reg_change = 0.0
            best_reg_val    = None
            best_dir        = None

            for direction_target, linspace_vals in [
                ("increase", np.linspace(orig_num, feat_max, n_steps)[1:]),
                ("decrease", np.linspace(orig_num, feat_min, n_steps)[1:]),
            ]:
                for step_val in linspace_vals:
                    test_row = original_row.copy()
                    test_row[feat] = float(step_val)
                    try:
                        X_cf     = _transform_row(test_row, original_columns, encoding_map,
                                                   preprocessor, feature_names)
                        new_pred = model.predict(X_cf)[0]

                        if is_classification and new_pred != orig_pred:
                            flip_val  = round(float(step_val), 4)
                            direction = direction_target
                            break
                        elif not is_classification:
                            orig_reg_v = float(model.predict(X_orig)[0])
                            new_reg_v  = float(new_pred)
                            delta      = abs(new_reg_v - orig_reg_v)
                            if delta > best_reg_change:
                                best_reg_change = delta
                                best_reg_val    = round(float(step_val), 4)
                                best_dir        = direction_target
                                best_new_pred   = new_reg_v
                    except Exception:
                        continue

                if is_classification and flip_val is not None:
                    break

            if is_classification and flip_val is not None:
                change_amount = abs(flip_val - orig_num)
                change_pct    = (change_amount / abs(orig_num) * 100) if orig_num != 0 else 0
                arrow         = "⬆️" if direction == "increase" else "⬇️"
                plain = (
                    f"{arrow} <b>{feat}</b>: change from <b>{orig_num:.4g}</b> "
                    f"to <b>{flip_val:.4g}</b> "
                    f"({'increase' if direction == 'increase' else 'decrease'} "
                    f"of {change_amount:.4g}) to flip the prediction."
                )
                suggestions.append({
                    "feature":          feat,
                    "original_value":   orig_num,
                    "suggested_value":  flip_val,
                    "change_direction": direction,
                    "change_amount":    round(change_amount, 4),
                    "change_pct":       round(change_pct, 1),
                    "feature_type":     "numeric",
                    "plain_english":    plain,
                })

            elif not is_classification and best_reg_val is not None:
                arrow = "⬆️" if best_dir == "increase" else "⬇️"
                plain = (
                    f"{arrow} <b>{feat}</b>: changing from <b>{orig_num:.4g}</b> "
                    f"to <b>{best_reg_val:.4g}</b> shifts the prediction from "
                    f"<b>{float(model.predict(X_orig)[0]):.4g}</b> "
                    f"to <b>{best_new_pred:.4g}</b> "
                    f"(Δ {best_reg_change:.4g})."
                )
                suggestions.append({
                    "feature":          feat,
                    "original_value":   orig_num,
                    "suggested_value":  best_reg_val,
                    "change_direction": best_dir,
                    "change_amount":    round(best_reg_change, 4),
                    "change_pct":       None,
                    "feature_type":     "numeric",
                    "plain_english":    plain,
                })

    # Sort classification by smallest % change; regression by largest impact
    if is_classification:
        suggestions.sort(key=lambda x: x.get("change_pct") or 0)
    else:
        suggestions.sort(key=lambda x: -(x.get("change_amount") or 0))

    return suggestions[:top_n_features]