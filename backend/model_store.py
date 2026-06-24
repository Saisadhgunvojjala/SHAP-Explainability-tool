"""
backend/model_store.py — Save and load trained models with all artifacts.

Saves everything needed to make predictions without retraining:
  - Trained model
  - SHAP explainer
  - Preprocessor (fitted ColumnTransformer)
  - Feature names
  - Encoding map (for high-cardinality columns)
  - Label encoder (if target was categorical)
  - Metadata (accuracy, task, dataset info, timestamp)

Why save the preprocessor?
  Without it, new data entered by the client can't be transformed the same
  way as training data — different scaling, different encoding, wrong predictions.
"""

import os
import json
import joblib
from datetime import datetime

SAVE_DIR = "saved_models"


def save_model(
    model_name: str,
    model,
    explainer,
    preprocessor,
    feature_names: list,
    original_columns: list,
    encoding_map: dict,
    label_encoder,
    task: str,
    target_name: str,
    metrics: dict,
    dataset_info: dict,
) -> str:
    """
    Save all model artifacts to saved_models/<model_name>/.
    Returns the save path.

    Args:
        model_name:       User-given name for this model (e.g. "diabetes_rf_v1")
        metrics:          dict with accuracy/f1/roc_auc or r2/mae/rmse
        dataset_info:     dict with n_rows, n_features, dataset_filename
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Sanitize name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in model_name)
    save_path = os.path.join(SAVE_DIR, safe_name)
    os.makedirs(save_path, exist_ok=True)

    # Save artifacts
    joblib.dump(model,        os.path.join(save_path, "model.pkl"))
    joblib.dump(explainer,    os.path.join(save_path, "explainer.pkl"))
    joblib.dump(preprocessor, os.path.join(save_path, "preprocessor.pkl"))

    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(save_path, "label_encoder.pkl"))

    with open(os.path.join(save_path, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    with open(os.path.join(save_path, "original_columns.json"), "w") as f:
        json.dump(original_columns, f)

    with open(os.path.join(save_path, "encoding_map.json"), "w") as f:
        json.dump(encoding_map, f)

    metadata = {
        "model_name":       safe_name,
        "task":             task,
        "target_name":      target_name,
        "feature_names":    feature_names,
        "original_columns": original_columns,
        "metrics":          metrics,
        "dataset_info":     dataset_info,
        "saved_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type":       type(model).__name__,
        "has_label_encoder": label_encoder is not None,
    }

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return save_path


def list_saved_models() -> list[dict]:
    """
    Return a list of saved model metadata dicts, sorted by save time (newest first).
    Returns empty list if no models saved yet.
    """
    if not os.path.exists(SAVE_DIR):
        return []

    models = []
    for name in os.listdir(SAVE_DIR):
        meta_path = os.path.join(SAVE_DIR, name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                models.append(json.load(f))

    return sorted(models, key=lambda x: x.get("saved_at", ""), reverse=True)


def load_model(model_name: str) -> dict:
    """
    Load all artifacts for a saved model.
    Returns dict with all components needed for prediction and explanation.

    Raises:
        FileNotFoundError if model_name doesn't exist.
    """
    save_path = os.path.join(SAVE_DIR, model_name)
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"No saved model found at '{save_path}'. "
            "Train and save a model first."
        )

    with open(os.path.join(save_path, "metadata.json")) as f:
        metadata = json.load(f)

    with open(os.path.join(save_path, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(save_path, "original_columns.json")) as f:
        original_columns = json.load(f)

    with open(os.path.join(save_path, "encoding_map.json")) as f:
        encoding_map = json.load(f)

    model        = joblib.load(os.path.join(save_path, "model.pkl"))
    explainer    = joblib.load(os.path.join(save_path, "explainer.pkl"))
    preprocessor = joblib.load(os.path.join(save_path, "preprocessor.pkl"))

    label_encoder = None
    le_path = os.path.join(save_path, "label_encoder.pkl")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)

    return {
        "model":            model,
        "explainer":        explainer,
        "preprocessor":     preprocessor,
        "label_encoder":    label_encoder,
        "feature_names":    feature_names,
        "original_columns": original_columns,
        "encoding_map":     encoding_map,
        "metadata":         metadata,
        "task":             metadata["task"],
        "target_name":      metadata["target_name"],
        "metrics":          metadata["metrics"],
    }


def delete_model(model_name: str):
    """Delete a saved model and all its artifacts."""
    import shutil
    save_path = os.path.join(SAVE_DIR, model_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
