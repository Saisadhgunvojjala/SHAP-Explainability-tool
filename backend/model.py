"""
backend/model.py — Production model registry with imbalance handling.

Real-world fix: imbalanced datasets (fraud, cybersecurity, rare diseases)
need class_weight="balanced" on tree/linear models, or SMOTE for severe imbalance.
Without this, a model on 99% negative class will just predict negative always
and still show 99% accuracy — which is useless.

Strategy:
- If minority class >= 10%  → class_weight="balanced" (fast, no data synthesis)
- If minority class <  10%  → SMOTE oversampling on training data before fitting
  (only on training set — never on test data, that would be leakage)
- Regression → no imbalance handling needed
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Threshold below which SMOTE is applied instead of class_weight
SMOTE_THRESHOLD = 0.10


def get_model_options(task: str) -> list:
    options = ["Random Forest", "Logistic Regression" if task == "Classification"
               else "Linear Regression", "Gradient Boosting", "SVM"]
    if XGBOOST_AVAILABLE:
        options.append("XGBoost")
    return options


def _apply_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    minority_ratio: float,
    is_imbalanced: bool,
) -> tuple:
    """
    Returns (X_train_final, y_train_final, strategy_used).
    Strategy used is a string for display in the UI.
    """
    if not is_imbalanced:
        return X_train, y_train, "None (balanced dataset)"

    if minority_ratio < SMOTE_THRESHOLD and SMOTE_AVAILABLE:
        # Severe imbalance → SMOTE
        k = min(5, int(y_train.value_counts().min()) - 1)
        if k < 1:
            return X_train, y_train, "class_weight=balanced (too few minority samples for SMOTE)"
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res),
            f"SMOTE (minority was {minority_ratio:.1%})",
        )
    else:
        # Moderate imbalance → class_weight (handled inside model instantiation)
        return X_train, y_train, f"class_weight=balanced (minority was {minority_ratio:.1%})"


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    model_name: str,
    is_imbalanced: bool = False,
    minority_ratio: float = 1.0,
):
    """
    Train a model with proper imbalance handling.

    Returns:
        model              — fitted sklearn-compatible model
        strategy_used      — string describing imbalance strategy applied
        X_train_used       — the X that was actually used for training
                             (may differ from X_train if SMOTE was applied)
    """
    use_balanced_weight = (
        is_imbalanced
        and minority_ratio >= SMOTE_THRESHOLD
        and task == "Classification"
    )

    # Apply SMOTE if needed (before model instantiation)
    X_train_used, y_train_used, strategy_used = _apply_imbalance_strategy(
        X_train, y_train, minority_ratio, is_imbalanced
    )

    # ── Model instantiation ───────────────────────────────────────────────────
    cw = "balanced" if use_balanced_weight else None

    if task == "Classification":
        if model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=200, class_weight=cw,
                random_state=42, n_jobs=-1
            )
        elif model_name == "Logistic Regression":
            model = LogisticRegression(
                max_iter=1000, class_weight=cw, random_state=42
            )
        elif model_name == "Gradient Boosting":
            # GBM doesn't support class_weight — SMOTE handles imbalance for it
            model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        elif model_name == "SVM":
            model = CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight=cw, random_state=42), cv=3
            )
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            # XGBoost uses scale_pos_weight for imbalance
            scale = ((1 - minority_ratio) / minority_ratio) if is_imbalanced else 1.0
            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale, tree_method="hist",
                random_state=42, eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    else:  # Regression
        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )
        elif model_name in ("Linear Regression", "Logistic Regression"):
            model = LinearRegression()
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        elif model_name == "SVM":
            model = SVR(kernel="rbf")
        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
            model = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                tree_method="hist", random_state=42,
            )
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    model.fit(X_train_used, y_train_used)
    return model, strategy_used, X_train_used
