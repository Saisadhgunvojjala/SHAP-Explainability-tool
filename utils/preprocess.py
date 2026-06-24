"""
utils/preprocess.py — Dataset-agnostic preprocessing pipeline.

Handles any CSV: mixed types, nulls, high-cardinality categoricals,
date strings, ID columns, class imbalance detection.

Returns a 15-tuple compatible with app.py's unpack.
dropped_columns is list[tuple[str,str]] — (col_name, reason) — so report.py
can iterate `for col, reason in dropped_columns` without crashing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import re
import warnings

warnings.filterwarnings("ignore")

# ── Thresholds ────────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SIZE        = 500
HIGH_CARDINALITY_THRESHOLD  = 20    # above → freq-encode
ID_COLUMN_THRESHOLD         = 0.95  # fraction unique → treat as ID, drop
NULL_DROP_THRESHOLD         = 0.60  # fraction null → drop column
MIN_ROWS                    = 30

# Characters XGBoost (and some sklearn models) reject in feature names
_BAD_CHARS_RE = re.compile(r'[\[\]<>{},.\s()#@!?/\\|]+')


def _sanitize_col(name: str) -> str:
    """Replace characters illegal in XGBoost feature names with underscores."""
    cleaned = _BAD_CHARS_RE.sub('_', str(name)).strip('_')
    return cleaned if cleaned else 'feature'


def _sanitize_columns(df) -> tuple:
    """
    Rename all columns to XGBoost-safe names.
    Returns (renamed_df, rename_map) where rename_map = {old: new}.
    """
    rename_map = {}
    seen       = {}
    for col in df.columns:
        new = _sanitize_col(col)
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        if new != col:
            rename_map[col] = new
    return df.rename(columns=rename_map), rename_map



def _is_datetime_col(series: pd.Series) -> bool:
    """Quick heuristic: try parsing first 20 non-null values as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def detect_and_drop_junk_columns(df: pd.DataFrame, target: str) -> tuple:
    """
    Drop columns that will hurt or are meaningless to the model.
    Returns (cleaned_df, dropped_columns) where dropped_columns is list[(name, reason)].
    """
    dropped = []
    to_drop = []

    for col in df.columns:
        if col == target:
            continue

        # Too many nulls
        null_frac = df[col].isnull().mean()
        if null_frac > NULL_DROP_THRESHOLD:
            to_drop.append(col)
            dropped.append((col, f"too many nulls ({null_frac:.0%})"))
            continue

        # Constant
        if df[col].nunique() <= 1:
            to_drop.append(col)
            dropped.append((col, "constant — zero variance"))
            continue

        # Datetime
        if _is_datetime_col(df[col]):
            to_drop.append(col)
            dropped.append((col, "datetime — dropped (no encoding without domain knowledge)"))
            continue

        # ID-like (high uniqueness ratio)
        # Only flag as ID for non-float columns — continuous floats are almost always
        # real features (temperature, salary, price) not row identifiers.
        # True IDs are strings (user_id, product_code) or integers with high cardinality.
        uniqueness = df[col].nunique() / max(len(df), 1)
        _is_float  = df[col].dtype in [np.float64, np.float32, float]
        if uniqueness >= ID_COLUMN_THRESHOLD and not _is_float:
            to_drop.append(col)
            dropped.append((col, f"ID-like ({uniqueness:.0%} unique values)"))
            continue

    df = df.drop(columns=to_drop)

    # ── Target leakage guard ──────────────────────────────────────────────────
    # Detect features that are directly derived from the target or near-identical.
    # Catches cases like Manufacturing: "Target" (binary) derived from "Failure Type" (multi-class)
    #
    # Three checks (any one triggers a drop):
    #   1. Pearson correlation >= 0.90 with target
    #   2. Feature unique values are a strict SUBSET of target values (binary flag from multi-class)
    #   3. Column name contains the target column name or "target" keyword (name heuristic)
    leakage_cols = []
    y_series  = df[target]
    y_vals    = set(y_series.dropna().unique())
    y_numeric = pd.to_numeric(y_series, errors="coerce")
    target_lower = target.lower()

    for col in df.columns:
        if col == target:
            continue

        reason = None

        # Check 1: high Pearson correlation
        try:
            x_numeric = pd.to_numeric(df[col], errors="coerce")
            valid     = x_numeric.notna() & y_numeric.notna()
            if valid.sum() >= 10 and x_numeric.notna().mean() > 0.5:
                corr = float(abs(x_numeric[valid].corr(y_numeric[valid])))
                if corr >= 0.90:
                    reason = f"target leakage — correlation {corr:.2f} with '{target}'"
        except Exception:
            pass

        # Check 2: unique values of feature are subset of target values
        # (e.g. Target={0,1} is subset of Failure_Type={0,1,2,3,4} → Target is derived)
        if reason is None:
            try:
                x_vals = set(df[col].dropna().unique())
                if len(x_vals) >= 2 and x_vals.issubset(y_vals) and len(x_vals) < len(y_vals):
                    reason = f"target leakage — values {sorted(x_vals)} are a subset of target values"
            except Exception:
                pass

        # Check 3: name heuristic — column name contains target name or "target"
        if reason is None:
            col_lower = col.lower()
            if (target_lower in col_lower or col_lower in target_lower or
                    col_lower in ("target", "label", "y") and col_lower != target_lower):
                # Only flag if it's also numeric (avoid dropping unrelated 'target_market' etc.)
                try:
                    x_numeric2 = pd.to_numeric(df[col], errors="coerce")
                    if x_numeric2.notna().mean() > 0.5:
                        reason = f"target leakage — column name '{col}' matches or contains target name"
                except Exception:
                    pass

        if reason:
            leakage_cols.append(col)
            dropped.append((col, reason))

    if leakage_cols:
        df = df.drop(columns=leakage_cols)

    return df, dropped


def detect_imbalance(y: pd.Series, task: str) -> tuple:
    """Returns (is_imbalanced, minority_ratio)."""
    if task != "Classification":
        return False, 1.0
    counts = y.value_counts(normalize=True)
    if len(counts) < 2:
        return False, 1.0
    minority_ratio = float(counts.min())
    return minority_ratio < 0.20, minority_ratio


def preprocess_data(df: pd.DataFrame, target: str):
    """
    Full preprocessing pipeline — works on any CSV with any column names.

    Returns 15-tuple:
        X_train, X_test, y_train, y_test,
        X_test_original, X_shap_background,
        preprocessor, feature_names, original_columns, task,
        is_imbalanced, minority_ratio, dropped_columns,
        encoding_map, label_encoder
    """
    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset too small ({len(df)} rows). Need at least {MIN_ROWS}.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    df = df.copy()

    # ── Drop rows with null target ────────────────────────────────────────────
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # ── Drop classes with only 1 sample (can't stratify-split them) ──────────
    class_counts = df[target].value_counts()
    rare = class_counts[class_counts < 2].index.tolist()
    if rare:
        df = df[~df[target].isin(rare)].reset_index(drop=True)

    # ── Encode categorical target ─────────────────────────────────────────────
    label_encoder = None
    if df[target].dtype == object or str(df[target].dtype) == "category":
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target].astype(str))

    # ── Auto-detect task ──────────────────────────────────────────────────────
    n_unique = df[target].nunique()
    task = "Classification" if n_unique <= 20 else "Regression"

    # ── Drop junk columns ─────────────────────────────────────────────────────
    df, dropped_columns = detect_and_drop_junk_columns(df, target)

    # ── Separate X / y ────────────────────────────────────────────────────────
    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing. "
                         "Check your dataset for all-null or all-unique columns.")

    # ── Sanitize column names (XGBoost rejects [, ], <, etc.) ─────────────────
    X, _col_rename_map = _sanitize_columns(X)
    # Update target column too if it was renamed (it's dropped from X but
    # we need original_columns to reflect sanitized names for inference)
    original_columns = X.columns.tolist()

    # ── Imbalance detection ───────────────────────────────────────────────────
    is_imbalanced, minority_ratio = detect_imbalance(y, task)

    # ── Train/test split ──────────────────────────────────────────────────────
    min_class = int(y.value_counts().min()) if task == "Classification" else 999
    stratify  = y if (task == "Classification" and min_class >= 2) else None
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)
    y_train     = y_train.reset_index(drop=True)
    y_test      = y_test.reset_index(drop=True)

    # ── Identify column types ─────────────────────────────────────────────────
    numeric_cols  = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
    cat_cols      = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card_cats = [c for c in cat_cols if X_train_raw[c].nunique() <= HIGH_CARDINALITY_THRESHOLD]
    high_card_cats= [c for c in cat_cols if X_train_raw[c].nunique() >  HIGH_CARDINALITY_THRESHOLD]

    # ── Frequency-encode high-cardinality categoricals ────────────────────────
    encoding_map = {}
    for col in high_card_cats:
        freq_map = X_train_raw[col].value_counts(normalize=True).to_dict()
        encoding_map[col] = freq_map
        X_train_raw[col]  = X_train_raw[col].map(freq_map).fillna(0).astype(float)
        X_test_raw[col]   = X_test_raw[col].map(freq_map).fillna(0).astype(float)

    numeric_cols_final = numeric_cols + high_card_cats
    cat_cols_final     = low_card_cats

    # ── Build sklearn pipeline ────────────────────────────────────────────────
    transformers = []
    if numeric_cols_final:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols_final))

    if cat_cols_final:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols_final))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr  = preprocessor.transform(X_test_raw)
    feature_names = numeric_cols_final + cat_cols_final

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_names)

    # ── SHAP background (stratified sample of train set) ──────────────────────
    bg_size = min(SHAP_BACKGROUND_SIZE, len(X_train))
    if bg_size >= len(X_train):
        X_shap_background = X_train.reset_index(drop=True)
    elif task == "Classification":
        from sklearn.model_selection import StratifiedShuffleSplit
        ratio = max(0.01, min(bg_size / len(X_train), 0.99))
        sss   = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
        _, idx = next(sss.split(X_train, y_train))
        X_shap_background = X_train.iloc[idx].reset_index(drop=True)
    else:
        X_shap_background = X_train.sample(n=bg_size, random_state=42).reset_index(drop=True)

    X_test_original = X_test_raw.copy()

    return (
        X_train, X_test,
        y_train, y_test,
        X_test_original, X_shap_background,
        preprocessor, feature_names, original_columns, task,
        is_imbalanced, minority_ratio, dropped_columns,
        encoding_map, label_encoder,
    )


def apply_inference_preprocessing(
    new_df: pd.DataFrame,
    original_columns: list,
    encoding_map: dict,
    preprocessor,
    feature_names: list,
) -> pd.DataFrame:
    """
    Re-apply saved preprocessing to new data at inference time.
    Raises ValueError with a clear message on schema mismatch.
    """
    missing = [c for c in original_columns if c not in new_df.columns]
    if missing:
        raise ValueError(
            f"New dataset is missing {len(missing)} required column(s): {missing}\n"
            f"This model expects: {original_columns}"
        )

    # Select and rename columns to match sanitized training names
    # original_columns already contains sanitized names; map from raw df
    # Try direct match first, then fall back to sanitized match
    col_map = {}
    for san_col in original_columns:
        if san_col in new_df.columns:
            col_map[san_col] = san_col
        else:
            # Find raw column that sanitizes to this name
            for raw_col in new_df.columns:
                if _sanitize_col(raw_col) == san_col:
                    col_map[raw_col] = san_col
                    break

    df = new_df[[c for c in col_map]].rename(columns=col_map).copy()

    # Apply frequency encoding (same maps from training)
    for col, freq_map in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].map(freq_map).fillna(0).astype(float)

    # Coerce any remaining object columns to numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    transformed = preprocessor.transform(df)
    return pd.DataFrame(transformed, columns=feature_names)