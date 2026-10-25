"""
app.py - Streamlit dashboard for the ML-based Network Intrusion Detection project.

Run with:   streamlit run app.py

IMPORTANT HONESTY NOTES (also shown in the app)
------------------------------------------------
* The app analyses UPLOADED CSV RECORDS. It does not capture or monitor live traffic.
* A prediction is a statistical guess from a model trained on one research dataset.
  It is not proof that an attack happened, and it will miss some attacks.
* All metrics shown come from reports/metrics.json written by train_model.py.
  If that file does not exist, the app shows instructions instead of numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.data_utils import (CATEGORICAL_FEATURES, EXCLUDED_FROM_MODEL, FEATURE_COLUMNS, MAX_UPLOAD_MB,
                            MAX_UPLOAD_ROWS, NUMERIC_FEATURES, TEST_FILE_NAME, TRAIN_FILE_NAME,
                            DataValidationError, load_and_validate, sanitize_for_csv_export)
from src.model_utils import (DATASET_SUMMARY_FILE, METRICS_FILE, MODEL_FILE, load_json_report,
                             load_pipeline, predict_traffic)

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
DATASET_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

# Colour palette: dark navy with restrained blue accents
NAVY, PANEL, BLUE, TEXT, MUTED = "#0B1220", "#111B2E", "#3B82F6", "#E5ECF6", "#8FA3BF"
C_NORMAL, C_MALICIOUS, C_UNCERTAIN = "#3B82F6", "#F87171", "#FBBF24"

logger = logging.getLogger("nids_app")

st.set_page_config(page_title="ML Intrusion Detection", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------
# Styling (static CSS only; no user-provided text is ever inserted into HTML)
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{ background-color: {NAVY}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid #1E2B45; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: 0.2px; }}
h1 {{ font-weight: 700; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #1E2B45; border-radius: 12px;
    padding: 14px 16px; }}
div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
.banner {{ background: {PANEL}; border: 1px solid #1E2B45; border-left: 4px solid {BLUE};
    border-radius: 10px; padding: 14px 18px; margin: 8px 0 18px 0; color: {TEXT}; }}
.subtle {{ color: {MUTED}; font-size: 0.92rem; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def show_chart(fig: go.Figure) -> None:
    """Apply the dark theme and draw a Plotly chart (works across Streamlit versions)."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT), margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def banner(text: str) -> None:
    st.markdown(f'<div class="banner">{text}</div>', unsafe_allow_html=True)  # text is always our own


def model_file_mtime() -> float:
    path = MODELS_DIR / MODEL_FILE
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner="Loading trained model ...")
def get_model(_mtime: float):
    """Load the model once. The mtime argument makes the cache refresh after re-training."""
    return load_pipeline(MODELS_DIR)


@st.cache_data(show_spinner=False)
def get_report(file_name: str, _mtime: float):
    return load_json_report(REPORTS_DIR, file_name)


def report_mtime(file_name: str) -> float:
    path = REPORTS_DIR / file_name
    return path.stat().st_mtime if path.exists() else 0.0


def get_metrics() -> dict | None:
    return get_report(METRICS_FILE, report_mtime(METRICS_FILE))


def get_dataset_summary() -> dict | None:
    return get_report(DATASET_SUMMARY_FILE, report_mtime(DATASET_SUMMARY_FILE))


def not_trained_help() -> None:
    st.warning("No trained model was found yet, so there is nothing to analyse or evaluate.")
    st.markdown(f"""
**How to fix this**
1. Download the two official CSV files (see the *Dataset information* page or the README):
   `{TRAIN_FILE_NAME}` and `{TEST_FILE_NAME}`.
2. Put both files in the project's `data/` folder.
3. In the VS Code terminal run: `python train_model.py`
4. Come back here and refresh the page.
""")


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(model, metadata, metrics) -> None:
    st.title("🛡️ Network Intrusion Detection Dashboard")
    st.markdown('<p class="subtle">Machine-learning classification of network traffic records '
                '(UNSW-NB15 dataset)</p>', unsafe_allow_html=True)
    banner("This tool analyses <b>uploaded CSV records</b> with a trained model. It does <b>not</b> monitor live "
           "network traffic, and its output is a <b>prediction</b>, not a verified attack.")

    if model is None or metrics is None:
        not_trained_help()
        return

    sel = metrics["selected_model"]
    sel_info = metrics["models"][sel]
    t = sel_info["test"]
    st.subheader("Selected model")
    st.write(f"**{sel_info['description']}** — trained on {metadata['training_rows']:,} records "
             f"on {metadata['trained_at_utc'][:10]}.")
    st.caption("Scores below were measured on the official UNSW-NB15 testing file, which was not used for training.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", pct(t["accuracy"]))
    c2.metric("Precision", pct(t["precision"]))
    c3.metric("Recall", pct(t["recall"]))
    c4.metric("F1-score", pct(t["f1"]))
    c5.metric("ROC-AUC", pct(t.get("roc_auc")))

    st.subheader("Latest analysis in this session")
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("No file analysed yet. Open **Traffic analysis** to upload a CSV.")
    else:
        res = analysis["results"]
        n_mal = int((res["prediction"] == "Predicted malicious").sum())
        a, b, c = st.columns(3)
        a.metric("Records analysed", f"{len(res):,}")
        b.metric("Predicted malicious", f"{n_mal:,}")
        c.metric("Uncertain predictions", f"{int((res['confidence'] == 'Uncertain').sum()):,}")


def page_dataset(metrics) -> None:
    st.title("📚 Dataset information")
    st.markdown(f"""
This project uses the **UNSW-NB15** dataset created by the Australian Centre for Cyber Security (UNSW Canberra).
Official page: [{DATASET_URL}]({DATASET_URL})
""")
    st.subheader("Files required")
    show_table(pd.DataFrame({
        "File": [TRAIN_FILE_NAME, TEST_FILE_NAME],
        "Purpose": ["Training the model", "Final held-out evaluation (never used for training)"],
        "Where to put it": ["data/ folder", "data/ folder"],
    }))
    st.markdown("""
Both files are in the official **“Training and Testing Sets”** download (a folder of partition files). They have a
header row. The four raw files (`UNSW-NB15_1.csv` … `_4.csv`) have **no header** and a different column layout, so
this app does **not** accept them rather than guessing what each column means.
""")

    summary = get_dataset_summary()
    if summary is None:
        st.info("Dataset statistics appear here after you run `python train_model.py` with the dataset in `data/`.")
    else:
        st.subheader("Class distribution (from the files used for training)")
        cols = st.columns(2)
        for col, key, title in ((cols[0], "training_file", "Training file"), (cols[1], "testing_file", "Testing file")):
            s = summary.get(key, {})
            lc = s.get("label_counts")
            with col:
                st.metric(f"{title}: records", f"{s.get('rows', 0):,}")
                if lc:
                    fig = px.pie(names=["Normal (0)", "Attack (1)"], values=[lc["normal"], lc["attack"]],
                                 hole=0.55, color_discrete_sequence=[C_NORMAL, C_MALICIOUS], title=title)
                    show_chart(fig)
        st.subheader("Attack categories (training file)")
        cats = summary.get("training_file", {}).get("attack_cat_counts")
        if cats:
            df = pd.DataFrame({"category": list(cats.keys()), "records": list(cats.values())}).sort_values("records")
            fig = px.bar(df, x="records", y="category", orientation="h", color_discrete_sequence=[BLUE])
            show_chart(fig)
            st.caption("The attack category is shown for information only. It is never used as a model input.")

    with st.expander("Which columns does the model use?"):
        st.markdown(f"**{len(FEATURE_COLUMNS)} input features** — categorical: {', '.join(CATEGORICAL_FEATURES)}")
        st.markdown("Numeric: " + ", ".join(NUMERIC_FEATURES))
        st.markdown(f"**Excluded from the model:** {', '.join(EXCLUDED_FROM_MODEL)} "
                    "(the answer, its category, and a row counter).")


def compare_with_labels(pred: pd.Series, labels: pd.Series) -> dict:
    y_pred = (pred == "Predicted malicious").astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(labels, y_pred),
            "precision": precision_score(labels, y_pred, zero_division=0),
            "recall": recall_score(labels, y_pred, zero_division=0),
            "f1": f1_score(labels, y_pred, zero_division=0),
            "cm": [[tn, fp], [fn, tp]]}


def run_analysis(uploaded, model) -> None:
    """Validate the upload, predict, and keep the results in memory (never on disk)."""
    try:
        with st.spinner("Validating the file ..."):
            data = load_and_validate(uploaded, require_label=False)
        with st.spinner(f"Predicting {len(data.features):,} records ..."):
            preds = predict_traffic(model, data.features)
    except DataValidationError as exc:
        st.session_state.pop("analysis", None)
        st.error(f"This file cannot be analysed. {exc}")
        return
    except Exception:
        logger.exception("Unexpected error during analysis")  # details go to the terminal only
        st.session_state.pop("analysis", None)
        st.error("An unexpected problem occurred while analysing this file. Please check that it is a "
                 "UNSW-NB15 training/testing-format CSV. Technical details were written to the terminal.")
        return

    f = data.features
    results = pd.DataFrame({"row": range(1, len(f) + 1)})
    if data.row_ids is not None:
        results["id"] = data.row_ids.values
    for col in ["proto", "service", "state", "dur", "sbytes", "dbytes"]:
        results[col] = f[col].values
    results = pd.concat([results, preds], axis=1)
    if data.labels is not None:
        results["file_label"] = data.labels.values  # the label that came WITH the file, for comparison only
    if data.attack_cat is not None:
        results["file_attack_cat"] = data.attack_cat.values

    st.session_state["analysis"] = {
        "results": results, "warnings": data.warnings, "filename_ok": True,
        "label_comparison": compare_with_labels(preds["prediction"], data.labels) if data.labels is not None else None,
    }
    st.success(f"Analysed {len(results):,} records. Open **Prediction results** in the sidebar to explore them.")
    for w in data.warnings:
        st.warning(w)


def page_analysis(model) -> None:
    st.title("🔎 Traffic analysis")
    banner("Upload a CSV file of traffic records in the official UNSW-NB15 training/testing format. "
           "Files are read in memory only and are <b>not saved</b>.")
    if model is None:
        not_trained_help()
        return

    st.markdown(f"""
**Requirements:** `.csv` file, up to {MAX_UPLOAD_MB} MB and {MAX_UPLOAD_ROWS:,} rows, with a header row and the
{len(FEATURE_COLUMNS)} UNSW-NB15 feature columns. Columns `id`, `label` and `attack_cat` are optional; the model never
uses them as inputs. If `label` is present, it is used only to compare against the predictions.
""")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], accept_multiple_files=False)
    col1, col2 = st.columns([1, 5])
    run = col1.button("Analyse", type="primary", disabled=uploaded is None)
    if col2.button("Clear results"):
        st.session_state.pop("analysis", None)
        st.info("Results cleared.")

    if run and uploaded is not None:
        run_analysis(uploaded, model)
    elif st.session_state.get("analysis"):
        st.info("Results from your last analysis are available on the **Prediction results** page.")


def page_results() -> None:
    st.title("📋 Prediction results")
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("No results yet. Upload and analyse a file on the **Traffic analysis** page first.")
        return

    res: pd.DataFrame = analysis["results"]
    banner("<b>These are model predictions, not verified attacks.</b> A “malicious” prediction means the record "
           "resembles attack traffic in the training data. It needs human review before any conclusion is drawn. "
           "Normal predictions can also be wrong.")

    n = len(res)
    n_mal = int((res["prediction"] == "Predicted malicious").sum())
    n_unc = int((res["confidence"] == "Uncertain").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analysed", f"{n:,}")
    c2.metric("Predicted malicious", f"{n_mal:,}", f"{n_mal / n * 100:.1f}% of records", delta_color="off")
    c3.metric("Predicted normal", f"{n - n_mal:,}")
    c4.metric("Uncertain", f"{n_unc:,}", f"{n_unc / n * 100:.1f}% of records", delta_color="off")
    if n_unc:
        st.warning(f"{n_unc:,} record(s) have a score close to the decision boundary (between 0.35 and 0.65). "
                   "Treat these predictions with extra caution.")

    left, right = st.columns(2)
    with left:
        counts = res["prediction"].value_counts().reindex(["Predicted normal", "Predicted malicious"]).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index, title="Prediction counts",
                     color_discrete_map={"Predicted normal": C_NORMAL, "Predicted malicious": C_MALICIOUS},
                     labels={"x": "", "y": "Records"})
        fig.update_layout(showlegend=False)
        show_chart(fig)
    with right:
        fig = px.histogram(res, x="malicious_probability", nbins=40, title="Distribution of model scores",
                           color_discrete_sequence=[BLUE], labels={"malicious_probability": "Malicious score (0–1)"})
        fig.add_vrect(x0=0.35, x1=0.65, fillcolor=C_UNCERTAIN, opacity=0.15, line_width=0,
                      annotation_text="uncertain", annotation_position="top left")
        show_chart(fig)

    left, right = st.columns(2)
    with left:
        proto = res.groupby("proto")["prediction"].apply(lambda s: (s == "Predicted malicious").sum())
        proto = proto.sort_values(ascending=False).head(10)
        if proto.sum() > 0:
            fig = px.bar(x=proto.index, y=proto.values, title="Predicted malicious records by protocol (top 10)",
                         labels={"x": "Protocol", "y": "Records"}, color_discrete_sequence=[C_MALICIOUS])
            show_chart(fig)
    with right:
        traffic = res["proto"].value_counts().head(10)
        fig = px.bar(x=traffic.index, y=traffic.values, title="Traffic by protocol (top 10)",
                     labels={"x": "Protocol", "y": "Records"}, color_discrete_sequence=[BLUE])
        show_chart(fig)

    cmp_ = analysis.get("label_comparison")
    if cmp_:
        st.subheader("Comparison with the labels included in your file")
        st.caption("Your file contained a `label` column, so the predictions can be compared with it. "
                   "This is only meaningful if those labels are correct and the file was not used to train the model.")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", pct(cmp_["accuracy"]))
        b.metric("Precision", pct(cmp_["precision"]))
        c.metric("Recall", pct(cmp_["recall"]))
        d.metric("F1-score", pct(cmp_["f1"]))
        cm = cmp_["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Predicted normal", "Predicted malicious"],
                        y=["Actually normal", "Actually attack"], color_continuous_scale="Blues",
                        title="Confusion matrix for this file")
        show_chart(fig)

    st.subheader("Results table")
    f1, f2 = st.columns(2)
    pred_filter = f1.selectbox("Show predictions", ["All", "Predicted malicious", "Predicted normal"])
    conf_filter = f2.selectbox("Show confidence", ["All", "Uncertain only", "Clearer only"])
    view = res
    if pred_filter != "All":
        view = view[view["prediction"] == pred_filter]
    if conf_filter == "Uncertain only":
        view = view[view["confidence"] == "Uncertain"]
    elif conf_filter == "Clearer only":
        view = view[view["confidence"] == "Clearer"]
    view = view.sort_values("malicious_probability", ascending=False)
    st.caption(f"{len(view):,} matching records (showing at most the first 1,000, highest scores first).")
    show_table(view.head(1000))

    csv_bytes = sanitize_for_csv_export(view).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered results as CSV", data=csv_bytes,
                       file_name="prediction_results.csv", mime="text/csv")
    st.caption("The downloaded file contains the rows currently selected by the filters above.")


def page_performance(metrics) -> None:
    st.title("📈 Model performance")
    if metrics is None:
        not_trained_help()
        return

    st.caption("All values were measured by train_model.py. The model was chosen using a validation split of the "
               "training file; the official testing file was used only for the final measurement.")
    models = metrics["models"]
    rows = []
    for name, info in models.items():
        t, v = info["test"], info["validation"]
        rows.append({"Model": info["description"] + (" ✅ selected" if name == metrics["selected_model"] else ""),
                     "Validation F1": round(v["f1"], 4), "Test accuracy": round(t["accuracy"], 4),
                     "Test precision": round(t["precision"], 4), "Test recall": round(t["recall"], 4),
                     "Test F1": round(t["f1"], 4),
                     "Test ROC-AUC": None if t.get("roc_auc") is None else round(t["roc_auc"], 4),
                     "Train time (s)": info["train_seconds"]})
    table = pd.DataFrame(rows)
    show_table(table)

    long = []
    for name, info in models.items():
        for m in ("accuracy", "precision", "recall", "f1"):
            long.append({"Model": info["description"].split(" (")[0], "Metric": m.capitalize(),
                         "Score": info["test"][m]})
    fig = px.bar(pd.DataFrame(long), x="Metric", y="Score", color="Model", barmode="group",
                 title="Test-set metrics by model", color_discrete_sequence=[BLUE, "#7DD3FC", "#94A3B8"])
    fig.update_yaxes(range=[0, 1])
    show_chart(fig)

    names = list(models.keys())
    chosen = st.selectbox("Inspect model", names, index=names.index(metrics["selected_model"]),
                          format_func=lambda n: models[n]["description"])
    info = models[chosen]
    t = info["test"]
    left, right = st.columns(2)
    with left:
        cm = t["confusion_matrix"]
        fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                        x=["Predicted normal", "Predicted attack"], y=["Actually normal", "Actually attack"],
                        color_continuous_scale="Blues", title="Confusion matrix (official test file)")
        show_chart(fig)
        st.caption(f"False positive rate: {pct(t.get('false_positive_rate'))} — normal traffic wrongly flagged. "
                   f"False negative rate: {pct(t.get('false_negative_rate'))} — attacks that were missed.")
    with right:
        if t.get("roc_curve"):
            fig = go.Figure()
            fig.add_scatter(x=t["roc_curve"]["fpr"], y=t["roc_curve"]["tpr"], mode="lines",
                            name=f"ROC (AUC = {t['roc_auc']:.3f})", line=dict(color=BLUE, width=3))
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                            line=dict(color="#64748B", dash="dash"))
            fig.update_layout(title="ROC curve (official test file)", xaxis_title="False positive rate",
                              yaxis_title="True positive rate")
            show_chart(fig)

    imp = info.get("feature_importances")
    if imp:
        df = pd.DataFrame(imp).sort_values("importance")
        fig = px.bar(df, x="importance", y="feature", orientation="h", title="Top feature importances",
                     color_discrete_sequence=[BLUE])
        show_chart(fig)
        st.caption("Importance shows which features the trees used most. It does not prove cause and effect.")

    st.markdown("**How to read these results:** the testing file comes from the same research dataset and lab "
                "environment as the training file. Good scores here do not guarantee equally good results on "
                "other networks, newer attacks or real production traffic.")


def page_about() -> None:
    st.title("ℹ️ About the project")
    st.markdown(f"""
**Machine-Learning-Based Network Intrusion Detection System** is an academic project that classifies network
traffic *records* as normal or potentially malicious using the public
[UNSW-NB15 dataset]({DATASET_URL}).

**How it works**
1. Official training and testing CSV files are validated.
2. A scikit-learn pipeline imputes missing values, scales numeric features and one-hot encodes categorical ones.
   Preprocessing is learned from training data only.
3. A Logistic Regression baseline and a Random Forest are compared on a validation split; the better one is saved.
4. The saved model scores the CSV records you upload.

**Technologies:** Python, Streamlit, pandas, NumPy, scikit-learn, Plotly, joblib, python-docx.

**Limitations**
- It analyses uploaded records; it does **not** monitor live network traffic.
- It does **not** detect all cyberattacks and will produce false alarms and missed attacks.
- It is trained on one dataset created in a laboratory setting and may not generalise to other networks.
- Predictions are not verified attacks; they should guide human investigation, not replace it.
- It is an educational project and must not be the only defence of a real system.
""")


# --------------------------------------------------------------------------
# Main: sidebar navigation
# --------------------------------------------------------------------------
def main() -> None:
    model, metadata = get_model(model_file_mtime())
    metrics = get_metrics()

    st.sidebar.title("🛡️ NIDS")
    page = st.sidebar.radio("Navigate", ["Dashboard overview", "Dataset information", "Traffic analysis",
                                         "Prediction results", "Model performance", "About the project"],
                            label_visibility="collapsed")
    st.sidebar.markdown("---")
    if metadata and metadata.get("demo_data"):
        st.error("⚠️ DEMO MODEL: this model was trained on RANDOMLY GENERATED data, not the real UNSW-NB15 dataset. "
                 "Every score and prediction is meaningless. Run `python train_model.py` with the official files "
                 "for real results.")
        st.sidebar.error("DEMO model (random data)")
    elif model is None:
        st.sidebar.warning("Model not trained")
    else:
        st.sidebar.success("Model loaded")
    st.sidebar.caption("Predictions are not verified attacks. Not a live monitoring tool.")

    if page == "Dashboard overview":
        page_overview(model, metadata, metrics)
    elif page == "Dataset information":
        page_dataset(metrics)
    elif page == "Traffic analysis":
        page_analysis(model)
    elif page == "Prediction results":
        page_results()
    elif page == "Model performance":
        page_performance(metrics)
    else:
        page_about()


main()
