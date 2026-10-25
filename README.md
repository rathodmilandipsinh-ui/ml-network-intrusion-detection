# Machine-Learning-Based Network Intrusion Detection System

An academic project that classifies **network traffic records** as *normal* or *potentially malicious*
using machine learning and the public **UNSW-NB15** dataset, with an interactive Streamlit dashboard.

> **Important:** this tool analyses CSV records that you upload. It does **not** monitor live network
> traffic, it does **not** detect all attacks, and its output is a *prediction*, not a verified attack.

## Objectives and features

- Validate and load the official UNSW-NB15 training and testing CSV files.
- Train and compare a lightweight baseline (Logistic Regression) with a Random Forest.
- Keep the official testing file completely separate from training and model selection.
- Save the fitted preprocessing + model pipeline with joblib.
- Streamlit dashboard with six sections: overview, dataset information, traffic analysis (CSV upload),
  prediction results (with CSV download), model performance and about.
- Defensive handling of untrusted uploads (file type, size, columns, data types, missing values).

## Technologies

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly, joblib, python-docx (for the final report).

## Project structure

```
app.py               Streamlit dashboard
train_model.py       Trains, compares and evaluates the models
src/data_utils.py    CSV loading and validation
src/model_utils.py   Pipeline, evaluation, prediction helpers
tests/               Automated checks (use random synthetic data, not real results)
data/                Put the dataset CSV files here (not included)
models/              Saved model (created by train_model.py)
reports/             Measured metrics (created by train_model.py)
.streamlit/          Dark theme and upload limit
```

## Getting the dataset

The dataset is **not included** and is **not downloaded automatically** (it is large).

1. Open the official page: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
2. Follow the download link on that page and find the folder with the **training and testing partition files**
   (named like "UNSW-NB15 - CSV Files" → "a part of training and testing set"). The page layout may change, so look for these two file names.
3. Download these two files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
4. Copy both into this project's `data/` folder. The names must match exactly.

Do **not** use `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv`. Those raw files have no header row and a different
column layout. The app refuses them instead of guessing what each column means.

If you use the dataset in your report, cite the creators (Moustafa & Slay, *UNSW-NB15: a comprehensive data set for
network intrusion detection systems*, MilCIS 2015) and follow the licence terms on the official page.

## No access to the dataset yet? Use the demo mode

The official download may ask you to sign in or request access. While you sort that out (or to practise),
you can run the **whole app on randomly generated demo data**:

```
python train_model.py --demo
streamlit run app.py
```

Then upload `data/demo/demo_upload_sample.csv` on the *Traffic analysis* page. (`python make_demo_data.py` creates
the demo files without training.)

**Demo data is NOT real network traffic and NOT UNSW-NB15.** Every score from it is meaningless, and the app shows a
red "DEMO MODEL" warning on every page while a demo model is loaded. Do **not** put demo scores in your report.
Training later with `python train_model.py` on the official files replaces the demo model and removes the warning.

## Setup on Windows (VS Code terminal)

Python 3.10 – 3.12 is recommended. Open the project folder in VS Code, open a terminal (**Terminal → New Terminal**) and run:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Your prompt should now start with `(.venv)`.)

## Train the model

After the two CSV files are in `data/`:

```
python train_model.py
```

This loads the data, compares the two models, measures them on the official testing file and saves:

- `models/ids_pipeline.joblib` and `models/model_metadata.json`
- `reports/metrics.json` and `reports/dataset_summary.json`

Optional flags if your computer is slow or low on memory:

```
python train_model.py --skip-random-forest        # baseline only
python train_model.py --max-train-rows 50000      # train on a random sample
```

## Launch the dashboard

```
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. To try the upload feature, upload a CSV in the same format
as the official files (for example the first few thousand rows of the testing file). Press `Ctrl+C` in the terminal to stop.

## Run the automated checks (optional)

```
python -m unittest discover -s tests -v
```

These use randomly generated data to prove the code runs; they say nothing about real detection quality.

## How the machine learning works (short version)

- **Target:** the `label` column (0 = normal, 1 = attack).
- **Inputs:** 42 traffic features. `id`, `label` and `attack_cat` are never given to the model
  (the last two would leak the answer).
- **Preprocessing:** numeric → median fill + standardise; categorical (`proto`, `service`, `state`) → most-frequent fill +
  one-hot encoding. All of it is learned from training data only, inside one scikit-learn `Pipeline`.
- **Class imbalance:** both models use balanced class weights.
- **Model choice:** a 20% validation split of the training file picks the better model (by F1). The official testing
  file is only used for the final measurement.
- **Reproducibility:** random seed 42.
- **Uncertain predictions:** malicious scores between 0.35 and 0.65 are flagged *Uncertain*. These scores are not
  guaranteed to be calibrated probabilities.

## Troubleshooting

| Problem | What to do |
|---|---|
| `python` is not recognised | Install Python from python.org and tick "Add Python to PATH", then reopen VS Code. |
| `.venv\Scripts\activate` is blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use **Command Prompt** instead. |
| `streamlit` is not recognised | Activate the virtual environment first, or run `python -m streamlit run app.py`. |
| Dataset site asks for login / access | Use the demo mode above for now. For real results ask your supervisor or university library for the two partition CSV files, or try the link from the official page again in a different browser. Only trust copies whose files are named exactly as above and have the 45-column header. |
| "Dataset files not found" | Check both CSV files are in `data/` with the exact official names. |
| Dashboard says "No trained model" | Run `python train_model.py`, then refresh the browser page. |
| Training runs out of memory / is very slow | Use `--skip-random-forest` or `--max-train-rows 50000`. Close other programs. |
| Upload rejected: columns missing / no header | Use a file in the official training/testing format (header row, 42 feature columns). |
| Upload rejected: file too large | Limit is 50 MB / 300,000 rows. Upload a smaller sample. |
| Model fails to load after upgrading packages | Re-run `python train_model.py` so the model is rebuilt with your installed scikit-learn version. |

## Limitations

- Offline analysis of uploaded CSV records only; **no live traffic capture or monitoring**.
- Does not detect every attack type and will produce both false alarms and missed attacks.
- Trained on a single laboratory-generated dataset; performance on other networks or newer attacks is unknown.
- The official testing file comes from the same source as the training file, so measured scores are likely
  optimistic compared with real-world use.
- Uploads must follow the UNSW-NB15 partition format; other formats are rejected.
- Predictions are not verified attacks and need human review.
- Only binary classification (normal vs attack) is implemented; attack-type classification is out of scope.
- The saved `.joblib` model is a pickle file; only load model files you created yourself.
- Developed and tested on Linux with current library versions; not yet run on Windows by the author of this code.

## Next step: the project report

After training, the measured numbers in `reports/metrics.json` can be used to write the final Word report
(python-docx). Ask for the report only once you have real results.
