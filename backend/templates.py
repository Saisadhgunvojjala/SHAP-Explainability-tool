"""
backend/templates.py — Industry-specific presets.

Each template defines:
- Recommended model
- Target column name hint
- Key features to watch
- Plain-English context shown in the UI
- SHAP interpretation hints specific to that industry
"""

TEMPLATES = {
    "💳 Credit Card Fraud": {
        "industry":      "Finance",
        "description":   "Detect fraudulent transactions. Target is typically a binary fraud flag (0/1). Highly imbalanced — genuine fraud is rare.",
        "target_hint":   ["fraud", "is_fraud", "Class", "label", "Fraud"],
        "model":         "XGBoost",
        "key_features":  ["amount", "hour", "distance_from_home", "merchant_category"],
        "shap_context":  "High SHAP on 'amount' means transaction size is the strongest signal. Negative SHAP on familiar merchant categories means known merchants reduce fraud risk.",
        "watch_for":     "Class imbalance — fraud is typically <1% of transactions. SMOTE will be applied automatically.",
        "metric_focus":  "ROC-AUC and F1 — accuracy is misleading on imbalanced fraud data.",
    },
    "🏦 Loan Approval": {
        "industry":      "Banking",
        "description":   "Predict whether a loan application will be approved or defaulted.",
        "target_hint":   ["loan_status", "default", "approved", "Loan_Status", "TARGET"],
        "model":         "Random Forest",
        "key_features":  ["income", "credit_score", "loan_amount", "employment_years", "debt_to_income"],
        "shap_context":  "Credit score and income typically dominate. High SHAP on debt_to_income ratio suggests overleveraged applicants are being correctly flagged.",
        "watch_for":     "Fairness — check if protected attributes (age, gender) appear in top SHAP features.",
        "metric_focus":  "F1 and ROC-AUC — false negatives (approving bad loans) are costly.",
    },
    "🏥 Healthcare / Diagnosis": {
        "industry":      "Healthcare",
        "description":   "Classify patient risk or diagnose conditions from clinical data.",
        "target_hint":   ["outcome", "Outcome", "diagnosis", "disease", "target", "label"],
        "model":         "Random Forest",
        "key_features":  ["glucose", "bmi", "age", "blood_pressure", "insulin"],
        "shap_context":  "Glucose and BMI are typically the strongest predictors for metabolic conditions. High SHAP on age indicates age-related risk progression.",
        "watch_for":     "Missing values in clinical data — zeros in biological columns (glucose, BMI) are often missing data, not real zeros.",
        "metric_focus":  "ROC-AUC — clinical tools need high sensitivity (catching true positives).",
    },
    "🔐 Cybersecurity / Intrusion": {
        "industry":      "Security",
        "description":   "Detect network intrusions, anomalies, or malicious traffic.",
        "target_hint":   ["label", "attack", "intrusion", "malicious", "class", "Category"],
        "model":         "XGBoost",
        "key_features":  ["duration", "protocol_type", "src_bytes", "dst_bytes", "flag"],
        "shap_context":  "Protocol type and byte counts are strong signals. High SHAP on 'duration' for short bursts may indicate port scanning.",
        "watch_for":     "Multi-class targets (attack types). The tool will use the class with highest mean |SHAP| for global explanation.",
        "metric_focus":  "F1 and ROC-AUC — false negatives (missed attacks) are critical.",
    },
    "👥 HR Attrition": {
        "industry":      "Human Resources",
        "description":   "Predict which employees are likely to leave the company.",
        "target_hint":   ["Attrition", "attrition", "left", "churned", "turnover"],
        "model":         "Random Forest",
        "key_features":  ["age", "job_satisfaction", "monthly_income", "years_at_company", "overtime"],
        "shap_context":  "Overtime and job satisfaction are typically top predictors. High SHAP on 'years_at_company' can indicate retention cliff points.",
        "watch_for":     "Ethical use — SHAP results for HR should not be used to discriminate. Validate that protected attributes are not top features.",
        "metric_focus":  "F1 — both false positives (wasted retention spend) and false negatives (losing good employees) matter.",
    },
    "🛍️ Customer Churn": {
        "industry":      "Retail / Telecom",
        "description":   "Predict which customers will stop using the product or service.",
        "target_hint":   ["churn", "Churn", "churned", "cancelled", "left"],
        "model":         "XGBoost",
        "key_features":  ["tenure", "monthly_charges", "contract_type", "num_products", "last_login_days"],
        "shap_context":  "Contract type and tenure are usually dominant. Short-tenure + month-to-month contracts = high churn risk.",
        "watch_for":     "Recency features (days_since_last_purchase) often have high SHAP but low interpretability without business context.",
        "metric_focus":  "F1 and ROC-AUC — early identification of at-risk customers is the goal.",
    },
    "🏠 Real Estate Pricing": {
        "industry":      "Real Estate",
        "description":   "Predict property sale price from features like size, location, and condition.",
        "target_hint":   ["price", "SalePrice", "sale_price", "value", "Price"],
        "model":         "XGBoost",
        "key_features":  ["sqft", "bedrooms", "location", "year_built", "condition"],
        "shap_context":  "Location and square footage dominate in most markets. High SHAP on year_built indicates buyers penalise older properties heavily.",
        "watch_for":     "Regression task — check R² and MAE rather than accuracy.",
        "metric_focus":  "R² and MAE — how close are predictions to actual sale prices.",
    },
    "⚙️ Manufacturing Defects": {
        "industry":      "Manufacturing",
        "description":   "Predict product defects or machine failures from sensor and process data.",
        "target_hint":   ["defect", "failure", "quality", "pass_fail", "anomaly"],
        "model":         "Random Forest",
        "key_features":  ["temperature", "pressure", "vibration", "speed", "humidity"],
        "shap_context":  "Temperature and pressure spikes typically signal defect risk. High SHAP on vibration may indicate bearing wear.",
        "watch_for":     "Sensor data often has outliers and noise — preprocessing handles median imputation but extreme outliers may need domain-specific treatment.",
        "metric_focus":  "F1 — missing a defect (false negative) is more costly than a false alarm.",
    },
    "📊 Custom / Other": {
        "industry":      "General",
        "description":   "Upload any CSV dataset. The tool will auto-detect task type and handle preprocessing.",
        "target_hint":   [],
        "model":         "Random Forest",
        "key_features":  [],
        "shap_context":  "No industry-specific context available. SHAP values show relative feature importance for your specific dataset.",
        "watch_for":     "Check that your target column is correctly identified and that ID columns are auto-dropped.",
        "metric_focus":  "Classification: F1 + ROC-AUC. Regression: R² + MAE.",
    },
}


def get_template_names() -> list:
    return list(TEMPLATES.keys())


def get_template(name: str) -> dict:
    return TEMPLATES.get(name, TEMPLATES["📊 Custom / Other"])


def suggest_target(df_columns: list, template_name: str) -> str:
    """
    Given a list of column names and a template, suggest the most likely target column.
    Returns the best match or the first column if nothing matches.
    """
    template = get_template(template_name)
    hints    = [h.lower() for h in template["target_hint"]]

    for col in df_columns:
        if col.lower() in hints:
            return col

    # Fuzzy match — check if any hint is a substring
    for col in df_columns:
        for hint in hints:
            if hint in col.lower() or col.lower() in hint:
                return col

    return df_columns[-1]  # fallback: last column (common convention)
