"""
app.py — SHAP Explainability Tool v3
Features: Auth · Save/Load · PDF · Industry Templates · Model Comparison · Batch Excel Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error

from utils.preprocess import preprocess_data
from backend.model import train_model, get_model_options
from backend.model_store import save_model, list_saved_models, load_model, delete_model
from backend.report import generate_pdf_report
from backend.templates import get_template_names, get_template, suggest_target
def _read_uploaded_file(f, zip_file_choice: str = None) -> pd.DataFrame:
    """
    Read any uploaded file into a DataFrame.
    Handles: CSV (any encoding), Excel (.xlsx/.xls), TSV, ZIP archives.

    For ZIP files: if multiple data files exist inside, zip_file_choice selects which one.
    Falls back through multiple encodings before raising.
    """
    import zipfile, io

    name = f.name.lower()

    # ── ZIP archive ───────────────────────────────────────────────────────────
    if name.endswith(".zip"):
        f.seek(0)
        with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
            # Find all readable data files inside
            data_exts = (".csv", ".xlsx", ".xls", ".tsv", ".txt")
            members   = [m for m in zf.namelist()
                         if any(m.lower().endswith(e) for e in data_exts)
                         and not m.startswith("__MACOSX")]
            if not members:
                raise ValueError(
                    "No CSV or Excel files found inside the ZIP. "
                    "Make sure the ZIP contains at least one .csv or .xlsx file."
                )
            # Use chosen file or first found
            target_member = zip_file_choice if zip_file_choice in members else members[0]
            with zf.open(target_member) as inner:
                inner_bytes = io.BytesIO(inner.read())
                inner_bytes.name = target_member   # mimic file object
                # Recurse with inner file
                return _read_uploaded_file(inner_bytes, zip_file_choice=None)

    # ── Excel ─────────────────────────────────────────────────────────────────
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)

    # ── TSV / TXT ─────────────────────────────────────────────────────────────
    if name.endswith((".tsv", ".txt")):
        for enc in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
            try:
                f.seek(0)
                return pd.read_csv(f, sep="\t", encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Could not read TSV/TXT: {e}")

    # ── CSV — try multiple encodings ──────────────────────────────────────────
    for enc in ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]:
        try:
            f.seek(0)
            return pd.read_csv(f, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")

    raise ValueError(
        "Could not decode the file. Try saving it as UTF-8 CSV from Excel "
        "(File → Save As → CSV UTF-8)."
    )


def _get_zip_members(f) -> list:
    """Return list of data file names inside a ZIP, or empty list if not a ZIP."""
    import zipfile, io
    if not f.name.lower().endswith(".zip"):
        return []
    try:
        f.seek(0)
        data_exts = (".csv", ".xlsx", ".xls", ".tsv", ".txt")
        with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
            return [m for m in zf.namelist()
                    if any(m.lower().endswith(e) for e in data_exts)
                    and not m.startswith("__MACOSX")]
    except Exception:
        return []


from backend.compare import run_comparison, plot_metrics_comparison, plot_shap_comparison, compute_agreement
from backend.export import generate_excel_export
from backend.confidence import get_confidence, batch_confidence_summary, regression_confidence
from backend.whatif import whatif_predict, find_counterfactual
from backend.fairness import full_fairness_report
from backend.explain_text import (
    global_explanation_text, local_explanation_text,
    render_explanation_card, render_summary_box,
)
from backend.shap_explainer import (
    get_shap_explainer, compute_shap_values, compute_single_row_shap,
    get_local_shap, global_shap_summary, plot_global_shap, plot_local_shap,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="SHAP Explainability Tool", page_icon="🔍")

st.markdown("""
<style>
.prediction-card{background:#f8f9fa;padding:20px;border-radius:12px;text-align:center;border:1px solid #e0e0e0;margin:20px 0}
.prediction-card h1{margin:0;font-size:32px;font-weight:600;color:#333}
.metric-card{background:#fff;padding:20px;border-radius:12px;text-align:center;border:1px solid #e0e0e0;margin:10px 0}
.metric-card h1{margin:10px 0 0 0;font-size:36px;font-weight:600;color:#2c3e50}
.explanation-box{padding:14px;margin-bottom:10px;background:#fff;border-radius:8px;border-left:4px solid;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.axis-label{background:#f8f9fa;padding:8px 15px;border-radius:6px;margin:6px 0;border-left:3px solid #667eea;font-size:14px}
.template-box{background:#f0f4ff;padding:16px;border-radius:10px;border:1px solid #c5d0f5;margin:10px 0}
.saved-card{background:#f0f4ff;padding:14px;border-radius:10px;border:1px solid #c5d0f5;margin-bottom:10px}
.warn-box{background:#fff8e1;border-left:4px solid #f9a825;padding:12px;border-radius:6px;margin:10px 0}
.stButton>button{background:#667eea;color:white;border:none;padding:10px 30px;border-radius:8px;width:100%}
</style>
""", unsafe_allow_html=True)

# ── Auth ──────────────────────────────────────────────────────────────────────

from backend.auth import login_page, logout_button
authenticator, name, auth_status, username = login_page()

if auth_status is False:
    st.error("Incorrect username or password.")
    st.stop()
if auth_status is None:
    st.warning("Please enter your username and password.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"### 👤 {name}")
    st.caption(f"@{username}")
    logout_button(authenticator, location="sidebar")
    st.divider()
    st.markdown("### 🗂️ Navigation")
    page = st.radio("Go to", [
        "🏠 Train New Model",
        "🔄 Compare Models",
        "📂 Load Saved Model",
        "📋 Manage Models",
    ], label_visibility="collapsed")

# ── Session defaults ──────────────────────────────────────────────────────────

DEFAULTS = {
    "stage":"input","model":None,"X_train":None,"X_test":None,
    "y_train":None,"y_test":None,"test_display":None,
    "feature_names":None,"task":None,"target_name":None,
    "preprocessor":None,"original_columns":None,"original_df":None,
    "encoding_map":None,"label_encoder":None,"y_pred":None,
    "shap_values":None,"shap_explainer":None,
    "is_imbalanced":False,"minority_ratio":1.0,
    "imbalance_strategy":"None","dropped_columns":[],
    "X_shap_background":None,"metrics":{},"dataset_info":{},"model_type_name":"",
    # compare state
    "cmp_results":None,"cmp_feature_names":None,
    # loaded model state
    "loaded_X_new":None,"loaded_preds":None,"loaded_shap_vals":None,
    "loaded_raw_df":None,"loaded_artifacts":None,"loaded_shap_size":0,
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k]=v


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN NEW MODEL
# ══════════════════════════════════════════════════════════════════════════════

if "Train" in page:

    if st.session_state.stage == "input":
        st.title("🔍 SHAP Explainability Tool")
        st.divider()

        # ── Industry template picker ──────────────────────────────────────────
        st.subheader("1️⃣ Choose Industry Template")
        template_name = st.selectbox("Industry", get_template_names())
        tmpl = get_template(template_name)

        st.markdown(f"""
<div class="template-box">
<b>🏭 {tmpl['industry']}</b> — {tmpl['description']}<br><br>
<b>⚠️ Watch for:</b> {tmpl['watch_for']}<br>
<b>📊 Focus metric:</b> {tmpl['metric_focus']}
</div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("2️⃣ Upload Dataset")
        file = st.file_uploader(
            "Upload CSV, Excel, or ZIP file",
            type=["csv", "xlsx", "xls", "txt", "zip"],
            help="Accepts .csv, .xlsx, .xls, .zip files. ZIP files containing CSVs or Excel files are supported."
        )
        zip_file_choice = None
        if file is not None and file.name.lower().endswith(".zip"):
            zip_members = _get_zip_members(file)
            if len(zip_members) > 1:
                zip_file_choice = st.selectbox(
                    "📂 Multiple files found in ZIP — select which to use:",
                    options=zip_members,
                )
            elif len(zip_members) == 1:
                zip_file_choice = zip_members[0]
                st.info(f"📂 Using file from ZIP: **{zip_file_choice}**")
            else:
                st.error("No CSV or Excel files found inside the ZIP.")
                file = None

        if file:
            df = _read_uploaded_file(file, zip_file_choice=zip_file_choice)
            st.success(f"✅ {df.shape[0]:,} rows · {df.shape[1]} columns")
            st.session_state.original_df = df.copy()

            with st.expander("👀 Preview"):
                st.dataframe(df.head(8), use_container_width=True)

            st.divider()
            st.subheader("3️⃣ Configure")
            c1, c2, c3 = st.columns(3)

            with c1:
                suggested = suggest_target(df.columns.tolist(), template_name)
                target_idx = df.columns.tolist().index(suggested) if suggested in df.columns else 0
                target = st.selectbox("🎯 Target Column", df.columns, index=target_idx)

            with c2:
                unique_count = df[target].nunique()
                auto_task    = "Classification" if unique_count <= 20 else "Regression"
                st.info(f"📋 Task: **{auto_task}**\n\n({unique_count} unique values)")

            with c3:
                default_model = tmpl["model"] if tmpl["model"] in get_model_options(auto_task) else get_model_options(auto_task)[0]
                model_idx     = get_model_options(auto_task).index(default_model)
                model_type    = st.selectbox("🤖 Model", get_model_options(auto_task), index=model_idx)

            # Class label configuration — only show for classification
            if auto_task == "Classification":
                target_vals = sorted(df[target].dropna().unique().tolist())
                st.divider()
                st.markdown("#### 🏷️ Class Labels *(optional but recommended)*")
                st.caption(
                    "Tell the tool what each class value means so SHAP charts show "
                    "'increases P(Good Credit)' instead of 'increases P(1)'."
                )
                label_cols = st.columns(len(target_vals))
                class_label_map = {}
                for i, val in enumerate(target_vals):
                    with label_cols[i]:
                        user_label = st.text_input(
                            f"Label for `{val}`",
                            value=str(val),
                            key=f"class_label_{val}",
                            help=f"Give a name to class {val} (e.g. 'Good Credit', 'Approved', 'Fraud')"
                        )
                        class_label_map[val] = user_label.strip() or str(val)

                # Positive class picker — which class should green bars point toward?
                positive_class_val = st.selectbox(
                    "🟢 Which class is the **positive / outcome of interest**?",
                    options=target_vals,
                    format_func=lambda v: f"{v} → {class_label_map.get(v, str(v))}",
                    index=len(target_vals)-1,  # default: last value (common convention)
                    help="SHAP green bars will mean 'increases probability of this class'. "
                         "For credit: pick the Bad/Default class so green = risk signal. "
                         "For fraud: pick the Fraud class. For approval: pick Approved."
                )
            else:
                class_label_map   = {}
                positive_class_val = None

            if tmpl["key_features"]:
                st.info(f"💡 Key features to watch for **{template_name}**: `{'`, `'.join(tmpl['key_features'])}`")

            if st.button("🚀 Train & Explain", type="primary"):
                with st.spinner("Preprocessing…"):
                    try:
                        (X_train,X_test,y_train,y_test,X_test_orig,X_shap_bg,
                         preprocessor,feature_names,original_columns,task,
                         is_imbalanced,minority_ratio,dropped_columns,
                         encoding_map,label_encoder) = preprocess_data(df, target)
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}"); st.stop()

                with st.spinner(f"Training {model_type}…"):
                    try:
                        model,strategy_used,X_train_used = train_model(
                            X_train,y_train,task,model_type,is_imbalanced,minority_ratio)
                    except Exception as e:
                        st.error(f"Training failed: {e}"); st.stop()

                with st.spinner("Computing SHAP…"):
                    try:
                        explainer   = get_shap_explainer(model, X_shap_bg)
                        shap_values = compute_shap_values(explainer, X_test)
                    except Exception as e:
                        st.error(f"SHAP failed: {e}"); st.stop()

                y_pred = model.predict(X_test)
                if task == "Classification":
                    acc = accuracy_score(y_test,y_pred)
                    f1  = f1_score(y_test,y_pred,average="weighted",zero_division=0)
                    try:
                        _proba = model.predict_proba(X_test)
                        _n_cls = _proba.shape[1]
                        if _n_cls == 2:
                            auc = round(float(roc_auc_score(y_test, _proba[:, 1])), 3)
                        else:
                            auc = round(float(roc_auc_score(y_test, _proba, multi_class="ovr", average="weighted")), 3)
                    except: auc = "N/A"
                    metrics = {"accuracy":round(acc,4),"f1":round(f1,4),"roc_auc":auc}
                else:
                    r2  = r2_score(y_test,y_pred)
                    mae = mean_absolute_error(y_test,y_pred)
                    rmse= float(np.sqrt(np.mean((np.array(y_test)-np.array(y_pred))**2)))
                    metrics = {"r2":round(r2,4),"mae":round(mae,4),"rmse":round(rmse,4)}

                test_display = X_test_orig.copy()
                test_display[target] = y_test.values

                # Build human-readable class names using user labels
                # LabelEncoder sorts classes — map encoded int back to user label
                if label_encoder is not None and class_label_map:
                    pos_name = class_label_map.get(positive_class_val, str(positive_class_val))
                    # Find which encoded int corresponds to positive_class_val
                    le_classes = label_encoder.classes_.tolist()
                    if positive_class_val in le_classes:
                        pos_encoded_idx = le_classes.index(positive_class_val)
                    else:
                        pos_encoded_idx = 1  # fallback
                    neg_vals = [v for v in le_classes if v != positive_class_val]
                    neg_name = class_label_map.get(neg_vals[0], str(neg_vals[0])) if neg_vals else "other"
                    # If user-chosen positive class is encoded as 0, we need to flip SHAP signs
                    shap_needs_flip = (pos_encoded_idx == 0)
                elif label_encoder is None and class_label_map:
                    # No label encoder (already numeric): find encoded index
                    all_vals = sorted(df[target].dropna().unique().tolist())
                    pos_encoded_idx = all_vals.index(positive_class_val) if positive_class_val in all_vals else 1
                    pos_name = class_label_map.get(positive_class_val, str(positive_class_val))
                    neg_vals = [v for v in all_vals if v != positive_class_val]
                    neg_name = class_label_map.get(neg_vals[0], str(neg_vals[0])) if neg_vals else "other"
                    shap_needs_flip = (pos_encoded_idx == 0)
                else:
                    pos_name        = str(positive_class_val) if positive_class_val is not None else "1"
                    neg_name        = "0"
                    shap_needs_flip = False

                # Flip SHAP values if user picked the class that LabelEncoder put at index 0
                if task == "Classification" and shap_needs_flip:
                    shap_values = -shap_values

                st.session_state.update({
                    "stage":"output","model":model,"X_train":X_train_used,
                    "X_test":X_test,"y_train":y_train,"y_test":y_test,
                    "test_display":test_display.reset_index(drop=True),
                    "feature_names":feature_names,"task":task,"target_name":target,
                    "preprocessor":preprocessor,"original_columns":original_columns,
                    "original_df":df,"encoding_map":encoding_map,"label_encoder":label_encoder,
                    "y_pred":y_pred,"shap_values":shap_values,"shap_explainer":explainer,
                    "is_imbalanced":is_imbalanced,"minority_ratio":minority_ratio,
                    "imbalance_strategy":strategy_used,"dropped_columns":dropped_columns,
                    "X_shap_background":X_shap_bg,"metrics":metrics,
                    "dataset_info":{"n_rows":len(df),"n_features":len(feature_names),"dataset_filename":file.name},
                    "model_type_name":model_type,"template_name":template_name,
                    "class_label_map":class_label_map,
                    "pos_class_name": pos_name,
                    "neg_class_name": neg_name,
                    "shap_was_flipped": task == "Classification" and shap_needs_flip,
                    "n_classes": len(model.classes_) if hasattr(model, "classes_") else 2,
                })
                st.rerun()

    elif st.session_state.stage in ("output","explanation"):
        model         = st.session_state.model
        X_test        = st.session_state.X_test
        y_test        = st.session_state.y_test
        y_pred        = st.session_state.y_pred
        test_display  = st.session_state.test_display
        feature_names = st.session_state.feature_names
        task          = st.session_state.task
        target_name   = st.session_state.target_name
        _label_enc    = st.session_state.get("label_encoder")
        # Use user-defined class names if set, otherwise fall back to encoded indices
        if task == "Classification":
            pos_class_name = st.session_state.get("pos_class_name") or (
                str(_label_enc.classes_[1]) if _label_enc is not None else "1"
            )
            neg_class_name = st.session_state.get("neg_class_name") or (
                str(_label_enc.classes_[0]) if _label_enc is not None else "0"
            )
        else:
            pos_class_name = target_name
            neg_class_name = ""
        preprocessor  = st.session_state.preprocessor
        original_columns = st.session_state.original_columns
        original_df   = st.session_state.original_df
        encoding_map  = st.session_state.encoding_map
        shap_values   = st.session_state.shap_values
        explainer     = st.session_state.shap_explainer
        metrics       = st.session_state.metrics
        is_imbalanced = st.session_state.is_imbalanced
        tmpl_name     = st.session_state.get("template_name","📊 Custom / Other")
        tmpl          = get_template(tmpl_name)

        # ── Shared decode helpers — available in ALL view branches ─────────────
        # Defined once here so What-If, Explanations, Fairness etc. can all use them
        # without redefining and without NameError when rendering persisted results.
        _le_disp    = st.session_state.get("label_encoder")
        _clabel_map = st.session_state.get("class_label_map", {})

        def _decode_label(encoded_val):
            """Decode an encoded integer back to original class label string."""
            try:
                if _le_disp is not None:
                    raw = _le_disp.inverse_transform([int(encoded_val)])[0]
                else:
                    raw = encoded_val
                return _clabel_map.get(raw, str(raw))
            except Exception:
                return str(encoded_val)

        c1,c2,c3,c4 = st.columns([2,2,2,2])
        with c1:
            if st.button("⬅️ Train New"):
                st.session_state.stage="input"; st.rerun()
        with c2:
            view = st.radio("View",["📊 Results","📖 Explanations","🎯 What-If","⚖️ Fairness","📦 Export"],
                            horizontal=True, label_visibility="collapsed")
        with c3:
            save_name = st.text_input("Save as",placeholder="model_name_v1",
                                      label_visibility="collapsed")
        with c4:
            if st.button("💾 Save Model"):
                if not save_name.strip():
                    st.error("Enter a name first.")
                else:
                    try:
                        save_model(save_name,model,explainer,preprocessor,feature_names,
                                   original_columns,encoding_map,st.session_state.label_encoder,
                                   task,target_name,metrics,st.session_state.dataset_info)
                        st.success(f"✅ Saved as **{save_name}**")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        st.divider()

        # ── Template SHAP context ─────────────────────────────────────────────
        if tmpl["shap_context"] and tmpl_name != "📊 Custom / Other":
            st.markdown(f'<div class="template-box">💡 <b>SHAP Insight for {tmpl_name}:</b> {tmpl["shap_context"]}</div>', unsafe_allow_html=True)

        # ── Results ───────────────────────────────────────────────────────────
        if "Results" in view:
            # ── Leakage / auto-removal warnings ──────────────────────────────
            _all_dropped = st.session_state.get("dropped_columns", [])
            _leakage_dropped = [(c, r) for c, r in _all_dropped if "leakage" in r]
            _routine_dropped = [(c, r) for c, r in _all_dropped if "leakage" not in r]

            if _leakage_dropped:
                st.error(
                    f"⚠️ **Target Leakage Detected & Fixed** — "
                    f"{len(_leakage_dropped)} column(s) were automatically removed before training "
                    f"because they are directly derived from your target (correlation ≥ 0.95). "
                    f"Keeping them would give the model a shortcut and make performance metrics "
                    f"misleadingly perfect on new data.\n\n" +
                    "\n".join(f"- **{c}** — {r}" for c, r in _leakage_dropped)
                )
            if _routine_dropped:
                with st.expander(f"🗑️ {len(_routine_dropped)} columns auto-removed (ID/constant/datetime)", expanded=False):
                    for col, reason in _routine_dropped:
                        st.markdown(f"- **{col}**: {reason}")

            st.subheader("📄 Test Dataset")
            st.dataframe(test_display, height=280, use_container_width=True)
            st.divider()
            st.subheader("📈 Performance")

            if task == "Classification":
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><h3>🎯 Accuracy</h3><h1>{metrics["accuracy"]:.2%}</h1></div>',unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>📐 F1</h3><h1>{metrics["f1"]:.3f}</h1></div>',unsafe_allow_html=True)
                auc_d = metrics["roc_auc"] if metrics["roc_auc"]=="N/A" else f'{float(metrics["roc_auc"]):.3f}'
                c3.markdown(f'<div class="metric-card"><h3>📊 ROC-AUC</h3><h1>{auc_d}</h1></div>',unsafe_allow_html=True)
                if is_imbalanced:
                    st.info(f"💡 {tmpl['metric_focus']}")
            else:
                c1,c2,c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><h3>📊 R²</h3><h1>{metrics["r2"]:.3f}</h1></div>',unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><h3>📉 MAE</h3><h1>{metrics["mae"]:.3f}</h1></div>',unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><h3>📏 RMSE</h3><h1>{metrics["rmse"]:.3f}</h1></div>',unsafe_allow_html=True)

        # ── Explanations ──────────────────────────────────────────────────────
        elif "Explanations" in view:
            summary  = global_shap_summary(shap_values, feature_names, top_n=10)
            exp_type = st.radio("Type",["🌍 Global","🔍 Local"],horizontal=True)
            st.divider()

            if st.button("📄 Download PDF Report"):
                with st.spinner("Generating…"):
                    try:
                        pdf = generate_pdf_report(
                            model_name=st.session_state.model_type_name,
                            task=task,target_name=target_name,metrics=metrics,
                            feature_names=feature_names,global_summary=summary,
                            dataset_info=st.session_state.dataset_info,
                            is_imbalanced=is_imbalanced,
                            imbalance_strategy=st.session_state.imbalance_strategy,
                            dropped_columns=st.session_state.dropped_columns,
                            username=username,
                        )
                        st.download_button("⬇️ Download PDF",pdf,
                                           f"shap_report_{target_name}.pdf","application/pdf")
                    except Exception as e:
                        st.error(f"PDF failed: {e}")

            st.divider()

            if "Global" in exp_type:
                st.subheader("🌍 Global Feature Importance")
                fig = plot_global_shap(summary, feature_names, positive_class=pos_class_name)
                st.pyplot(fig); plt.close(fig)
                _le = st.session_state.get("label_encoder")
                _pos_class = str(_le.classes_[1]) if (_le is not None and task == "Classification") else "prediction"
                _neg_class = str(_le.classes_[0]) if (_le is not None and task == "Classification") else ""
                _axis_label = (f"Green (right) = increases P(<b>{_pos_class}</b>) · Red (left) = increases P({_neg_class}) · % = share of influence"
                               if task == "Classification" else
                               "Green (right) = increases prediction · Red (left) = decreases · % = share of influence")
                st.markdown(f'<div class="axis-label">{_axis_label}</div>',unsafe_allow_html=True)

                show_tech_g = st.toggle("Show technical details", value=False, key="global_tech", help="Show SHAP values alongside plain-English explanations")
                st.markdown("### 🧠 What Drives This Model?")
                g_expls = global_explanation_text(
                    feature_names=feature_names, mean_abs=summary["mean_abs"],
                    mean_sign=summary["mean_sign"], X_test=X_test, task=task,
                    target_name=target_name, top_idx=summary["top_idx"], total=summary["total_abs"],
                )
                for exp in g_expls:
                    st.markdown(render_explanation_card(exp, show_technical=show_tech_g), unsafe_allow_html=True)

            else:
                st.subheader("🔍 Local Explanation")
                source = st.radio("Input",["📊 Use Test Data","✏️ Enter New Data"],horizontal=True)
                X_single=None; local_shap=None; current_pred=None; current_actual=None; local_row_data=None

                if source=="📊 Use Test Data":
                    row_num = st.number_input("Row",1,len(test_display),1)
                    st.dataframe(test_display.iloc[[row_num-1]],use_container_width=True)
                    X_single=X_test.iloc[row_num-1:row_num]
                    current_pred=model.predict(X_single)[0]
                    current_actual=y_test.iloc[row_num-1]
                    # Multi-class: recompute SHAP for predicted class, not the global class
                    _n_cls_local = len(model.classes_) if hasattr(model, "classes_") else 2
                    if task == "Classification" and _n_cls_local > 2:
                        _pred_cls = int(current_pred)
                        local_shap = compute_single_row_shap(explainer, X_single, predicted_class=_pred_cls)
                    else:
                        local_shap = get_local_shap(shap_values, row_num-1)
                    local_row_data=test_display.iloc[row_num-1].to_dict()
                else:
                    new_data={}
                    cols_ui=st.columns(2)
                    from utils.preprocess import _sanitize_col as _sc
                    _s2o = {_sc(c): c for c in original_df.columns}
                    _s2o.update({c: c for c in original_df.columns})
                    _col_idx = 0
                    for feat in original_columns:
                        if feat == target_name or _s2o.get(feat) == target_name:
                            continue
                        # Skip leakage columns
                        _dropped_en = st.session_state.get("dropped_columns", [])
                        _leakage_en = {c for c, r in _dropped_en if "leakage" in r}
                        _orig_en    = _s2o.get(feat, feat)
                        if feat in _leakage_en or _orig_en in _leakage_en:
                            continue
                        orig_col = _s2o.get(feat)
                        if orig_col is None or orig_col not in original_df.columns:
                            # Not in original_df — silently carry forward from test mean
                            try:
                                new_data[feat] = float(X_test[feat].mean()) if feat in X_test.columns else 0.0
                            except Exception:
                                new_data[feat] = 0.0
                            continue
                        col_data = original_df[orig_col].dropna()
                        with cols_ui[_col_idx % 2]:
                            is_cat = (col_data.dtype == object or
                                      str(col_data.dtype) == "category" or
                                      (col_data.nunique() <= 15 and col_data.dtype not in ["float64","float32"]))
                            if is_cat:
                                cats = sorted(col_data.unique().tolist(), key=str)
                                new_data[feat] = st.selectbox(f"📋 {feat}", [str(c) for c in cats], key=f"en_{feat}")
                            else:
                                col_min = float(col_data.min())
                                col_max = float(col_data.max())
                                col_mean= float(col_data.mean())
                                new_data[feat] = st.number_input(
                                    f"🔢 {feat}", value=col_mean,
                                    min_value=col_min, max_value=col_max,
                                    key=f"en_{feat}"
                                )
                        _col_idx += 1
                    if st.button("🔮 Predict & Explain"):
                        try:
                            from utils.preprocess import apply_inference_preprocessing
                            X_single = apply_inference_preprocessing(
                                pd.DataFrame([new_data]),
                                original_columns, encoding_map,
                                preprocessor, feature_names,
                            )
                            current_pred=model.predict(X_single)[0]
                            _pred_cls_idx = int(current_pred) if task == "Classification" else None
                            local_shap=compute_single_row_shap(explainer, X_single, predicted_class=_pred_cls_idx)
                            local_row_data=new_data
                        except Exception as e:
                            st.error(f"Failed: {e}")

                if X_single is not None and local_shap is not None:
                    # _le_disp, _clabel_map, _decode_label defined at top of output stage
                    if task == "Regression":
                        pred_disp = f"{float(current_pred):.2f}"
                    else:
                        pred_disp = _decode_label(current_pred)
                    if current_actual is not None:
                        actual_str = f"<p style='color:#666;margin:5px 0 0 0'>Actual: {_decode_label(current_actual)}</p>"
                    else:
                        actual_str = ""
                    st.markdown(f'<div class="prediction-card"><h3>🎯 Prediction</h3><h1>{pred_disp}</h1>{actual_str}</div>',unsafe_allow_html=True)

                    # Orientation banner — works for binary AND multi-class
                    if task == "Classification":
                        _enc_pred   = int(current_pred)
                        _n_classes  = len(_le_disp.classes_) if _le_disp is not None else 2
                        _pred_label = _decode_label(_enc_pred)
                        _prob_str   = ""
                        _prob_table = ""
                        try:
                            _all_probs = model.predict_proba(X_single)[0]
                            _prob = float(_all_probs[_enc_pred])
                            _prob_str = f" (confidence: {_prob:.0%})"
                            if _n_classes > 2:
                                # Show full probability breakdown for multi-class
                                _cls_labels = [_decode_label(i) for i in range(_n_classes)]
                                _prob_rows  = sorted(
                                    zip(_cls_labels, _all_probs),
                                    key=lambda x: -x[1]
                                )
                                _prob_table = " &nbsp;|&nbsp; ".join(
                                    f"**{lbl}**: {p:.0%}" for lbl, p in _prob_rows
                                )
                        except Exception:
                            pass

                        if _n_classes == 2:
                            st.info(
                                f"**🟢 Green = pushing toward '{pos_class_name}'** &nbsp;|&nbsp; "
                                f"**🔴 Red = pushing toward '{neg_class_name}'**\n\n"
                                f"Predicted **{_pred_label}**{_prob_str}. "
                                f"Green features supported this outcome; red features worked against it."
                            )
                        else:
                            # Multi-class: SHAP shows the predicted class vs all others
                            st.info(
                                f"**Predicted: {_pred_label}**{_prob_str}\n\n"
                                + (f"Class probabilities: {_prob_table}\n\n" if _prob_table else "")
                                + f"🟢 Green bars = features that pushed the model toward **{_pred_label}**. "
                                f"🔴 Red bars = features that pushed away from **{_pred_label}**."
                            )

                    # ── "How to get a different prediction" guide ──────────
                    _n_cls_exp  = len(_le_disp.classes_) if _le_disp is not None else 2
                    _pred_lbl_exp = _decode_label(int(current_pred))
                    _flip_label = neg_class_name if _n_cls_exp == 2 else "a different class"
                    with st.expander(f"💡 What inputs would produce a **'{_flip_label}'** prediction?", expanded=False):
                        st.markdown(
                            f"To push the model **away from {_pred_lbl_exp}**, "
                            f"try reversing the features currently showing **green**. "
                            f"Here's what to change based on this row's SHAP values:"
                        )
                        _local_arr  = local_shap if hasattr(local_shap, '__len__') else local_shap.tolist()
                        _local_np   = __import__('numpy').array(_local_arr).flatten()
                        _abs_sorted = __import__('numpy').argsort(__import__('numpy').abs(_local_np))[::-1]
                        _rows = []
                        for _idx in _abs_sorted[:6]:
                            _idx  = int(_idx)
                            _feat = feature_names[_idx]
                            _sv   = float(_local_np[_idx])
                            _raw  = (local_row_data or {}).get(_feat, "?")
                            if _sv > 0:
                                # Currently pushing toward pos_class — to flip, decrease this
                                _rows.append({
                                    "Feature": _feat,
                                    "Current Value": _raw,
                                    "Currently pushing toward": f"✅ {pos_class_name}",
                                    "To get Bad: try": f"⬇️ Lower / worse value of {_feat}",
                                })
                            else:
                                # Already pushing toward neg_class — reinforce it
                                _rows.append({
                                    "Feature": _feat,
                                    "Current Value": _raw,
                                    "Currently pushing toward": f"❌ {neg_class_name}",
                                    "To get Bad: try": f"Keep or worsen {_feat}",
                                })
                        import pandas as _pd2
                        st.dataframe(_pd2.DataFrame(_rows), use_container_width=True)
                        st.caption(
                            f"These are the top features by SHAP influence for this row. "
                            f"Features marked ✅ are currently helping predict {pos_class_name} — "
                            f"making them worse would increase the risk of {neg_class_name}."
                        )

                    fig=plot_local_shap(local_shap, feature_names, positive_class=pos_class_name)
                    st.pyplot(fig); plt.close(fig)
                    _le2 = st.session_state.get("label_encoder")
                    _pos2 = str(_le2.classes_[1]) if (_le2 is not None and task == "Classification") else "prediction"
                    _neg2 = str(_le2.classes_[0]) if (_le2 is not None and task == "Classification") else ""
                    _local_lbl = (f"Green = increases P(<b>{_pos2}</b>) · Red = increases P({_neg2}) · % = this row's share"
                                  if task == "Classification" else
                                  "Green = increases prediction · Red = decreases · % = this row's share")
                    st.markdown(f'<div class="axis-label">{_local_lbl}</div>',unsafe_allow_html=True)

                    show_tech_l = st.toggle("Show technical details", value=False, key="local_tech", help="Show SHAP values alongside plain-English explanations")
                    st.markdown("### 🧠 Why This Prediction?")
                    summary_txt, l_expls = local_explanation_text(
                        feature_names=feature_names, local_shap=local_shap,
                        raw_row=local_row_data or {}, prediction=current_pred,
                        task=task, target_name=target_name, X_train=st.session_state.get("X_train"),
                        positive_class=pos_class_name, negative_class=neg_class_name,
                    )
                    st.markdown(render_summary_box(summary_txt, current_pred, task, target_name, label_encoder=_le_disp), unsafe_allow_html=True)
                    st.markdown("#### Feature-by-Feature Breakdown")
                    for exp in l_expls:
                        st.markdown(render_explanation_card(exp, show_technical=show_tech_l), unsafe_allow_html=True)

                    if st.button("📄 Download PDF with Local Explanation"):
                        try:
                            summary2=global_shap_summary(shap_values,feature_names,top_n=10)
                            pdf=generate_pdf_report(
                                model_name=st.session_state.model_type_name,task=task,
                                target_name=target_name,metrics=metrics,feature_names=feature_names,
                                global_summary=summary2,dataset_info=st.session_state.dataset_info,
                                is_imbalanced=is_imbalanced,
                                imbalance_strategy=st.session_state.imbalance_strategy,
                                dropped_columns=st.session_state.dropped_columns,
                                local_shap=local_shap,local_row_data=local_row_data,
                                local_prediction=pred_disp,username=username,
                            )
                            st.download_button("⬇️ Download PDF",pdf,
                                               f"shap_local_{target_name}.pdf","application/pdf")
                        except Exception as e:
                            st.error(f"PDF failed: {e}")

        # ── What-If ───────────────────────────────────────────────────────────
        elif "What-If" in view:
            from backend.whatif import whatif_predict, find_counterfactual
            st.subheader("🎯 What-If Analysis")
            st.caption(
                "Modify feature values, run the prediction, and see counterfactuals — "
                "all visible at once. Works for any industry dataset."
            )

            # ── Row selector ─────────────────────────────────────────────────
            row_num = st.number_input(
                "Base row from test set", 1, len(test_display), 1, key="wi_row"
            )
            base_row_raw = test_display.iloc[row_num - 1].to_dict()
            base_row_raw.pop(target_name, None)

            with st.expander("📋 View base row data", expanded=False):
                st.dataframe(test_display.iloc[[row_num - 1]], use_container_width=True)

            st.divider()

            # ── Two-column layout: inputs LEFT, results RIGHT ─────────────────
            left_col, right_col = st.columns([1, 1], gap="large")

            with left_col:
                st.markdown("#### ✏️ Modify Feature Values")
                modified = {}

                # Build a reverse map: sanitized_name → original_df column name
                # so we can look up col_data even when names were sanitized
                from utils.preprocess import _sanitize_col
                _san_to_orig = {}
                for orig_col in original_df.columns:
                    san = _sanitize_col(orig_col)
                    _san_to_orig[san]     = orig_col   # sanitized → original
                    _san_to_orig[orig_col]= orig_col   # original  → original (identity)

                for feat in original_columns:
                    # Skip the target column — it must never appear as an input
                    if feat == target_name or _san_to_orig.get(feat) == target_name:
                        continue
                    # Skip leakage columns (detected by preprocess.py)
                    _dropped = st.session_state.get("dropped_columns", [])
                    _leakage_names = {c for c, r in _dropped if "leakage" in r}
                    _orig_feat = _san_to_orig.get(feat, feat)
                    if feat in _leakage_names or _orig_feat in _leakage_names:
                        continue

                    # Resolve to original_df column (handles sanitized names)
                    orig_col_name = _san_to_orig.get(feat)
                    if orig_col_name is None or orig_col_name not in original_df.columns:
                        # Column not in original_df — use default from base_row_raw
                        default_val = base_row_raw.get(feat)
                        if default_val is not None:
                            try:
                                modified[feat] = float(default_val)
                            except (TypeError, ValueError):
                                modified[feat] = str(default_val)
                        continue

                    col_data    = original_df[orig_col_name].dropna()
                    default_val = base_row_raw.get(feat, base_row_raw.get(orig_col_name))

                    is_cat = (
                        col_data.dtype == object
                        or str(col_data.dtype) == "category"
                        or (col_data.nunique() <= 15 and col_data.dtype not in ["float64", "float32"])
                    )

                    if is_cat:
                        cats = sorted(col_data.unique().tolist(), key=str)
                        try:
                            sel_idx = [str(c) for c in cats].index(str(default_val)) if default_val is not None else 0
                        except ValueError:
                            sel_idx = 0
                        modified[feat] = st.selectbox(
                            f"📋 {feat}", [str(c) for c in cats],
                            index=sel_idx, key=f"wi_{feat}"
                        )
                    else:
                        try:
                            dv = float(default_val) if default_val is not None else float(col_data.mean())
                        except (TypeError, ValueError):
                            dv = float(col_data.mean())
                        col_min = float(col_data.min())
                        col_max = float(col_data.max())
                        # Clamp default to valid range
                        dv = max(col_min, min(col_max, dv))
                        modified[feat] = st.number_input(
                            f"🔢 {feat}", value=dv,
                            min_value=col_min,
                            max_value=col_max,
                            key=f"wi_{feat}"
                        )

                run_btn = st.button("🔮 Run Prediction + Counterfactual", type="primary", use_container_width=True)

            # ── Results column — persisted in session state so they stay visible ──
            with right_col:
                # Initialise result stores
                if "wi_result"       not in st.session_state: st.session_state.wi_result       = None
                if "wi_suggestions"  not in st.session_state: st.session_state.wi_suggestions  = None
                if "wi_error"        not in st.session_state: st.session_state.wi_error        = None
                if "wi_pred_disp"    not in st.session_state: st.session_state.wi_pred_disp    = None
                if "wi_cf_error"     not in st.session_state: st.session_state.wi_cf_error     = None

                if run_btn:
                    # Run prediction
                    with st.spinner("Running prediction…"):
                        try:
                            wi_result = whatif_predict(
                                model=model, explainer=explainer, preprocessor=preprocessor,
                                encoding_map=encoding_map, feature_names=feature_names,
                                original_columns=original_columns, modified_values=modified, task=task,
                            )
                            if task == "Regression":
                                _pd = f"{float(wi_result['prediction']):.4g}"
                            else:
                                _pd = _decode_label(wi_result['prediction'])
                            st.session_state.wi_result    = wi_result
                            st.session_state.wi_pred_disp = _pd
                            st.session_state.wi_error     = None
                        except Exception as e:
                            st.session_state.wi_error  = str(e)
                            st.session_state.wi_result = None

                    # Run counterfactual immediately after prediction
                    with st.spinner("Finding counterfactuals…"):
                        try:
                            suggestions = find_counterfactual(
                                model=model, preprocessor=preprocessor,
                                encoding_map=encoding_map, feature_names=feature_names,
                                original_columns=original_columns, original_row=modified,
                                X_train=st.session_state.X_train, task=task,
                                original_df=original_df, explainer=explainer,
                            )
                            st.session_state.wi_suggestions = suggestions
                            st.session_state.wi_cf_error    = None
                        except Exception as e:
                            st.session_state.wi_cf_error    = str(e)
                            st.session_state.wi_suggestions = None

                # ── Always render results if available ────────────────────────
                if st.session_state.wi_error:
                    st.error(f"Prediction failed: {st.session_state.wi_error}")

                if st.session_state.wi_result is not None:
                    wi_r   = st.session_state.wi_result
                    pd_str = st.session_state.wi_pred_disp

                    st.markdown(
                        f'<div class="prediction-card" style="padding:14px;margin-bottom:12px;">' +
                        f'<h4 style="margin:0 0 4px 0;">🎯 Prediction</h4>' +
                        f'<h2 style="margin:0;">{pd_str}</h2></div>',
                        unsafe_allow_html=True
                    )

                    if wi_r.get("probability") is not None:
                        _prob    = wi_r["probability"]
                        _n_wi    = len(_le_disp.classes_) if _le_disp is not None else 2
                        if _n_wi == 2:
                            _pct   = _prob if _prob >= 0.5 else 1 - _prob
                            _pname = pos_class_name if _prob >= 0.5 else neg_class_name
                            st.metric("Confidence", f"{_pct:.1%}", help=f"Probability of {_pname}")
                        else:
                            # Multi-class: use all_probs stored directly in result
                            _all_p = wi_r.get("all_probs")
                            if _all_p is not None:
                                _cls = [_decode_label(i) for i in range(len(_all_p))]
                                _prob_df = pd.DataFrame({
                                    "Class": _cls,
                                    "Probability": [f"{p:.1%}" for p in _all_p]
                                }).sort_values("Probability", ascending=False)
                                st.dataframe(_prob_df, use_container_width=True, hide_index=True)
                            else:
                                st.metric("Confidence", f"{_prob:.1%}")

                    _wi_pred_cls = wi_r.get("pred_class")
                    _wi_pos_lbl  = (_decode_label(_wi_pred_cls)
                                    if task == "Classification" and _wi_pred_cls is not None
                                    else pos_class_name)
                    with st.expander(f"📊 SHAP Breakdown — why '{_wi_pos_lbl}'?", expanded=True):
                        _shap_vals = wi_r["shap_values"]
                        _shap_arr  = np.array(_shap_vals)
                        if np.abs(_shap_arr).sum() < 1e-10:
                            st.warning(
                                "SHAP values are near-zero for this prediction. "
                                "This typically means the model is extremely confident "
                                "and features have little marginal influence on this particular row."
                            )
                        else:
                            fig = plot_local_shap(_shap_vals, feature_names, positive_class=_wi_pos_lbl)
                            st.pyplot(fig); plt.close(fig)

                # ── Counterfactual section — always visible below prediction ──
                st.divider()
                _cf_label = "flip the outcome" if task == "Classification" else "shift the prediction most"
                st.markdown(f"#### 🔄 What needs to change to **{_cf_label}**?")

                if st.session_state.wi_cf_error:
                    st.error(f"Counterfactual search failed: {st.session_state.wi_cf_error}")
                elif st.session_state.wi_suggestions is None:
                    st.info("Run the prediction above to see counterfactuals.")
                elif len(st.session_state.wi_suggestions) == 0:
                    st.warning(
                        "No single-feature change found that flips this prediction. "
                        "This is a high-confidence case — the model is very certain. "
                        "Multiple features would need to change simultaneously."
                    )
                else:
                    _sug_color = "#667eea" if task == "Classification" else "#28a745"
                    for i, s in enumerate(st.session_state.wi_suggestions):
                        _icon = "🔢" if s.get("feature_type") == "numeric" else "📋"
                        _rank = f"#{i+1}"
                        # Industry-friendly change summary
                        if s.get("change_pct") is not None:
                            _delta = f" ({s['change_pct']}% change)"
                        elif s.get("change_amount") is not None:
                            _delta = f" (impact: {s['change_amount']:.4g})"
                        else:
                            _delta = ""
                        st.markdown(
                            f'<div class="explanation-box" style="border-left:4px solid {_sug_color};margin-bottom:8px;">' +
                            f'<b>{_icon} {_rank}</b>{_delta}<br>{s["plain_english"]}</div>',
                            unsafe_allow_html=True
                        )

        # ── Fairness ──────────────────────────────────────────────────────────
        elif "Fairness" in view:
            st.subheader("⚖️ Fairness & Bias Detection")
            if task != "Classification":
                st.info("Fairness metrics are currently available for classification tasks only.")
            else:
                g_summary_f = global_shap_summary(shap_values, feature_names, top_n=10)
                fairness = full_fairness_report(
                    df=original_df,
                    predictions=y_pred,
                    column_names=original_df.columns.tolist(),
                    shap_top_features=g_summary_f["top_features"],
                    actual_labels=y_test,
                    task=task,
                )

                detection = fairness["detection"]
                if detection["warning"]:
                    st.markdown(detection["warning"], unsafe_allow_html=True)
                else:
                    st.success("✅ No protected attributes detected in top SHAP features.")

                if fairness["group_results"]:
                    st.divider()
                    for col, result in fairness["group_results"].items():
                        st.markdown(f"#### Protected Attribute: `{col}`")
                        st.markdown(result["verdict"], unsafe_allow_html=True)

                        rows = [[s["group"], s["pred_pct"], s.get("tpr_pct","N/A"), str(s["n"])]
                                for s in result["group_stats"]]
                        df_fair = pd.DataFrame(rows, columns=["Group","Prediction Rate","True Positive Rate","Count"])
                        st.dataframe(df_fair, use_container_width=True)

                        c1, c2 = st.columns(2)
                        c1.metric("Disparate Impact Ratio", f"{result['disparate_impact']:.3f}",
                                  help="80% rule: value < 0.8 may indicate illegal discrimination")
                        c2.metric("Max Prediction Rate Gap", f"{result['disparity']:.1%}")
                        st.divider()
                elif fairness["has_protected"]:
                    st.info("Protected attributes found in dataset but not enough groups for statistical comparison.")
                else:
                    st.info("No protected-sounding column names detected in this dataset.")

        # ── Export ────────────────────────────────────────────────────────────
        else:
            st.subheader("📦 Batch Export — All Rows")
            st.markdown("Download SHAP explanations for every row as a formatted Excel workbook.")
            st.info(f"SHAP computed on {min(500,len(X_test)):,} of {len(X_test):,} test rows. All rows included in Predictions sheet.")

            if st.button("📊 Generate Excel Export", type="primary"):
                with st.spinner("Building Excel workbook…"):
                    try:
                        shap_size = min(500, len(X_test))
                        X_shap_export = X_test.iloc[:shap_size]
                        shap_export   = compute_shap_values(explainer, X_shap_export)

                        excel_bytes = generate_excel_export(
                            raw_df=test_display.iloc[:shap_size].reset_index(drop=True),
                            predictions=y_pred[:shap_size],
                            shap_values=shap_export,
                            feature_names=feature_names,
                            target_name=target_name,
                            shap_row_count=shap_size,
                        )
                        st.success(f"✅ Excel ready — {shap_size:,} rows with SHAP explanations")
                        st.download_button(
                            "⬇️ Download Excel",
                            excel_bytes,
                            f"shap_export_{target_name}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                        st.markdown("""
**Workbook contains 4 sheets:**
- **Predictions** — original data + predicted values
- **SHAP Values** — raw SHAP per feature per row (green=positive, red=negative)
- **Plain English Explanations** — top 3 reasons per row in plain language
- **Feature Summary** — global feature importance ranking
""")
                    except Exception as e:
                        st.error(f"Export failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPARE MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif "Compare" in page:
    st.title("🔄 Model Comparison")
    st.markdown("Train multiple models on the same dataset and compare performance + SHAP side by side.")
    st.divider()

    file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls", "txt"],
            key="cmp_upload",
            help="Accepts .csv, .xlsx, .xls files."
        )

    if file:
        df = _read_uploaded_file(file)
        st.success(f"✅ {df.shape[0]:,} rows · {df.shape[1]} columns")

        c1,c2 = st.columns(2)
        with c1:
            target = st.selectbox("🎯 Target Column", df.columns, key="cmp_target")
        with c2:
            unique_count = df[target].nunique()
            auto_task    = "Classification" if unique_count<=20 else "Regression"
            st.info(f"Task: **{auto_task}**")

        all_models    = get_model_options(auto_task)
        models_to_run = st.multiselect(
            "Select models to compare (2–4 recommended)",
            all_models,
            default=all_models[:3],
        )

        if len(models_to_run) < 2:
            st.warning("Select at least 2 models to compare.")
        elif st.button("🚀 Run Comparison", type="primary"):
            with st.spinner("Preprocessing…"):
                try:
                    (X_train,X_test,y_train,y_test,X_test_orig,X_shap_bg,
                     preprocessor,feature_names,original_columns,task,
                     is_imbalanced,minority_ratio,dropped_columns,
                     encoding_map,label_encoder) = preprocess_data(df, target)
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}"); st.stop()

            with st.spinner(f"Training {len(models_to_run)} models and computing SHAP…"):
                try:
                    cmp_results = run_comparison(
                        X_train,X_test,y_train,y_test,X_shap_bg,
                        feature_names,task,is_imbalanced,minority_ratio,
                        models_to_run,
                    )
                    st.session_state.cmp_results      = cmp_results
                    st.session_state.cmp_feature_names= feature_names
                except Exception as e:
                    st.error(f"Comparison failed: {e}"); st.stop()

    # ── Show comparison results ───────────────────────────────────────────────
    if st.session_state.cmp_results is not None:
        results       = st.session_state.cmp_results["results"]
        feature_names = st.session_state.cmp_feature_names
        valid         = [r for r in results if "error" not in r]
        errored       = [r for r in results if "error" in r]

        if errored:
            for r in errored:
                st.warning(f"⚠️ {r['model_name']} failed: {r['error']}")

        if valid:
            st.divider()
            st.subheader("📈 Performance Comparison")

            # Metrics table
            rows = []
            for r in valid:
                row = {"Model": r["model_name"]}
                row.update(r["metrics"])
                rows.append(row)
            metrics_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(metrics_df.style.highlight_max(axis=0, color="#c8e6c9")
                                         .highlight_min(axis=0, color="#ffcdd2"),
                         use_container_width=True)

            fig = plot_metrics_comparison(valid, task if 'task' in dir() else "Classification")
            if fig:
                st.pyplot(fig); plt.close(fig)

            st.divider()
            st.subheader("🔍 SHAP Feature Importance Comparison")
            st.caption("Which features does each model rely on most?")

            fig2 = plot_shap_comparison(valid, feature_names, top_n=8)
            if fig2:
                st.pyplot(fig2); plt.close(fig2)

            st.divider()
            st.subheader("🤝 Model Agreement")
            st.caption("Spearman correlation of feature importance rankings. 1.0 = models agree completely on which features matter.")

            agreement_df = compute_agreement(valid, feature_names)
            if agreement_df is not None:
                st.dataframe(agreement_df.style.background_gradient(cmap="RdYlGn",vmin=0,vmax=1),
                             use_container_width=True)
                st.info("💡 High agreement (>0.8) means models are consistent. Low agreement means different models are picking up different patterns — worth investigating.")

            # Best model recommendation
            st.divider()
            st.subheader("🏆 Recommendation")
            # Pick best metric: prefer ROC-AUC > F1 > Accuracy for classification,
            # R² for regression. Fall back to first available key.
            _PREFERRED = ["ROC-AUC", "F1", "Accuracy", "R²", "MAE", "RMSE"]
            _all_keys = list(valid[0]["metrics"].keys())
            key = next((k for k in _PREFERRED if k in _all_keys), _all_keys[0])
            # For MAE/RMSE lower is better — invert sign for max()
            _lower_is_better = key in ("MAE", "RMSE")
            def _score(r):
                v = r["metrics"].get(key, "N/A")
                if v == "N/A":
                    return float("-inf")
                return -float(v) if _lower_is_better else float(v)
            best = max(valid, key=_score)
            direction = "lowest" if _lower_is_better else "best"
            st.success(f"✅ **{best['model_name']}** has the {direction} **{key}** of `{best['metrics'].get(key)}`")
            # Show full ranking
            ranked = sorted(valid, key=_score, reverse=True)
            rank_rows = [[i+1, r["model_name"]] + [r["metrics"].get(k, "N/A") for k in _all_keys]
                         for i, r in enumerate(ranked)]
            import pandas as _pd
            rank_df = _pd.DataFrame(rank_rows, columns=["Rank", "Model"] + _all_keys)
            st.dataframe(rank_df.set_index("Rank"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LOAD SAVED MODEL
# ══════════════════════════════════════════════════════════════════════════════

elif "Load" in page:
    st.title("📂 Load Saved Model")
    st.divider()

    saved = list_saved_models()
    if not saved:
        st.info("No saved models yet. Train and save a model first.")
    else:
        model_names = [m["model_name"] for m in saved]
        selected    = st.selectbox("Select a saved model", model_names)
        meta        = next(m for m in saved if m["model_name"]==selected)

        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><h3>Task</h3><h1 style="font-size:20px">{meta["task"]}</h1></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h3>Model</h3><h1 style="font-size:20px">{meta["model_type"]}</h1></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><h3>Saved</h3><h1 style="font-size:16px">{meta["saved_at"]}</h1></div>',unsafe_allow_html=True)
        st.markdown("**Training Metrics:**"); st.json(meta["metrics"])
        st.caption(f"Required columns: `{', '.join(meta['original_columns'])}`")

        st.divider()
        new_file = st.file_uploader(
            "Upload CSV or Excel for prediction + explanation",
            type=["csv", "xlsx", "xls", "txt"],
            key="load_upload",
            help="Accepts .csv, .xlsx, .xls files."
        )

        if new_file:
            new_df_raw = _read_uploaded_file(new_file)
            st.success(f"✅ {len(new_df_raw):,} rows loaded")
            st.dataframe(new_df_raw.head(5), use_container_width=True)

            missing = [c for c in meta["original_columns"] if c not in new_df_raw.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if st.button("🚀 Run Predictions & Explain", type="primary"):
                    with st.spinner("Predicting and computing SHAP…"):
                        try:
                            from utils.preprocess import apply_inference_preprocessing

                            artifacts  = load_model(selected)
                            a_model    = artifacts["model"]
                            a_explainer= artifacts["explainer"]
                            a_prep     = artifacts["preprocessor"]
                            a_features = artifacts["feature_names"]
                            a_orig     = artifacts["original_columns"]
                            a_enc      = artifacts["encoding_map"]
                            a_task     = artifacts["task"]
                            a_target   = artifacts["target_name"]

                            try:
                                X_new = apply_inference_preprocessing(
                                    new_df_raw, a_orig, a_enc, a_prep, a_features
                                )
                            except ValueError as ve:
                                st.error(f"❌ Schema mismatch: {ve}")
                                st.stop()
                            preds  = a_model.predict(X_new)
                            shap_size = min(500,len(X_new))
                            shap_vals = compute_shap_values(a_explainer, X_new.iloc[:shap_size])

                            st.session_state.update({
                                "loaded_X_new":X_new,"loaded_preds":preds,
                                "loaded_shap_vals":shap_vals,"loaded_raw_df":new_df_raw,
                                "loaded_artifacts":artifacts,"loaded_shap_size":shap_size,
                            })
                        except Exception as e:
                            st.error(f"Failed: {e}")

                if st.session_state.loaded_preds is not None:
                    artifacts  = st.session_state.loaded_artifacts
                    X_new      = st.session_state.loaded_X_new
                    preds      = st.session_state.loaded_preds
                    shap_vals  = st.session_state.loaded_shap_vals
                    raw_df     = st.session_state.loaded_raw_df
                    a_features = artifacts["feature_names"]
                    a_task     = artifacts["task"]
                    a_target   = artifacts["target_name"]
                    shap_size  = st.session_state.loaded_shap_size

                    st.divider()
                    result_df = raw_df.copy()
                    result_df["Prediction"] = preds
                    st.subheader("📊 Predictions")
                    st.dataframe(result_df, use_container_width=True, height=280)
                    st.download_button("⬇️ Predictions CSV",
                                       result_df.to_csv(index=False).encode(),
                                       "predictions.csv","text/csv")

                    st.divider()

                    # Excel export for loaded model
                    if st.button("📊 Download Full Excel Export"):
                        with st.spinner("Building Excel…"):
                            try:
                                excel_bytes = generate_excel_export(
                                    raw_df=raw_df,predictions=preds[:shap_size],
                                    shap_values=shap_vals,feature_names=a_features,
                                    target_name=a_target,shap_row_count=shap_size,
                                )
                                st.download_button("⬇️ Download Excel",excel_bytes,
                                                   f"shap_export_{a_target}.xlsx",
                                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                            except Exception as e:
                                st.error(f"Export failed: {e}")

                    st.divider()
                    g_summary = global_shap_summary(shap_vals, a_features, top_n=10)
                    st.subheader("🌍 Global SHAP — New Data")
                    fig = plot_global_shap(g_summary, a_features)
                    st.pyplot(fig); plt.close(fig)

                    st.divider()
                    st.subheader("🔍 Row-Level Explanation")
                    row_num = st.number_input("Select row",1,shap_size,1)
                    st.dataframe(raw_df.iloc[[row_num-1]],use_container_width=True)

                    pred_val  = preds[row_num-1]
                    pred_disp = f"{float(pred_val):.2f}" if a_task=="Regression" else str(pred_val)
                    st.markdown(f'<div class="prediction-card"><h3>🎯 Prediction — Row {row_num}</h3><h1>{pred_disp}</h1></div>',unsafe_allow_html=True)

                    local_shap = get_local_shap(shap_vals, row_num-1)
                    fig = plot_local_shap(local_shap, a_features)
                    st.pyplot(fig); plt.close(fig)

                    show_tech_ld = st.toggle("Show technical details", value=False, key="load_tech", help="Show SHAP values alongside plain-English explanations")
                    st.markdown("### 🧠 Why This Prediction?")
                    l_summary_txt, l_expls = local_explanation_text(
                        feature_names=a_features, local_shap=local_shap,
                        raw_row=raw_df.iloc[row_num-1].to_dict(),
                        prediction=pred_val, task=a_task, target_name=a_target,
                    )
                    st.markdown(render_summary_box(l_summary_txt, pred_val, a_task, a_target), unsafe_allow_html=True)
                    st.markdown("#### Feature-by-Feature Breakdown")
                    for exp in l_expls:
                        st.markdown(render_explanation_card(exp, show_technical=show_tech_ld), unsafe_allow_html=True)

                    if st.button("📄 Download PDF for This Row"):
                        try:
                            pdf=generate_pdf_report(
                                model_name=selected,task=a_task,target_name=a_target,
                                metrics=artifacts["metrics"],feature_names=a_features,
                                global_summary=g_summary,
                                dataset_info={"n_rows":len(raw_df),"n_features":len(a_features)},
                                local_shap=local_shap,local_row_data=raw_df.iloc[row_num-1].to_dict(),
                                local_prediction=pred_disp,username=username,
                            )
                            st.download_button("⬇️ Download PDF",pdf,
                                               f"shap_row{row_num}.pdf","application/pdf")
                        except Exception as e:
                            st.error(f"PDF failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MANAGE MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif "Manage" in page:
    st.title("📋 Manage Saved Models")
    st.divider()
    saved = list_saved_models()
    if not saved:
        st.info("No saved models yet.")
    else:
        for m in saved:
            st.markdown(f"""
<div class="saved-card">
<b>🤖 {m['model_name']}</b> &nbsp;·&nbsp; {m['model_type']} &nbsp;·&nbsp;
Task: {m['task']} &nbsp;·&nbsp; Target: <code>{m['target_name']}</code><br>
<small>Saved: {m['saved_at']} &nbsp;·&nbsp;
Rows: {m.get('dataset_info',{}).get('n_rows','N/A'):,} &nbsp;·&nbsp;
Features: {m.get('dataset_info',{}).get('n_features','N/A')}</small>
</div>""",unsafe_allow_html=True)
            c1,c2=st.columns([5,1])
            with c2:
                if st.button("🗑️ Delete",key=f"del_{m['model_name']}"):
                    delete_model(m["model_name"])
                    st.success(f"Deleted {m['model_name']}")
                    st.rerun()