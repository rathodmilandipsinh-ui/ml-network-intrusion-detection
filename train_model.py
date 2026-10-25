"""
train_model.py - Train and evaluate the intrusion-detection models.

Run with:   python train_model.py

WHAT THIS SCRIPT DOES (step by step)
------------------------------------
1. Loads the official training and testing CSV files from the data/ folder.
2. Splits the TRAINING file into a smaller training part and a validation part.
   The validation part is used ONLY to choose between the candidate models.
3. Trains two candidates: Logistic Regression (baseline) and Random Forest.
4. Picks the candidate with the best validation F1 score.
5. Re-trains each candidate on the full training file and measures it ONCE on the
   official testing file. The testing file is never used for training or for choosing.
6. Saves the chosen pipeline (models/) and measured results (reports/).

Every number written to reports/ is computed from real data in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_utils import (DataValidationError, find_dataset_files, load_and_validate,
                            TRAIN_FILE_NAME, TEST_FILE_NAME)
from src.model_utils import (DATASET_SUMMARY_FILE, METRICS_FILE, MODEL_DESCRIPTIONS,
                             RANDOM_SEED, build_pipeline, evaluate_pipeline,
                             get_feature_importances, make_metadata, save_artifacts,
                             timed_fit)

PROJECT_DIR = Path(__file__).resolve().parent
CANDIDATES = ["logistic_regression", "random_forest"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the UNSW-NB15 intrusion detection models.")
    p.add_argument("--data-dir", default=str(PROJECT_DIR / "data"), help="Folder with the dataset CSVs.")
    p.add_argument("--models-dir", default=str(PROJECT_DIR / "models"), help="Where to save the model.")
    p.add_argument("--reports-dir", default=str(PROJECT_DIR / "reports"), help="Where to save metrics.")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="Optional: use only a random sample of the training rows (for quick tests / low memory).")
    p.add_argument("--demo", action="store_true",
                   help="Train on RANDOM demo data (data/demo) instead of the official files. "
                        "Scores are meaningless; use only to try the app.")
    p.add_argument("--skip-random-forest", action="store_true",
                   help="Train only the lightweight baseline (use this if your computer runs out of memory).")
    return p.parse_args()


def summarise_split(name: str, data) -> dict:
    """Class counts and missing values for the Dataset Information page."""
    summary = {"rows": int(len(data.features)),
               "missing_cells": int(data.features.isna().sum().sum())}
    if data.labels is not None:
        counts = data.labels.value_counts().to_dict()
        summary["label_counts"] = {"normal": int(counts.get(0, 0)), "attack": int(counts.get(1, 0))}
    if data.attack_cat is not None:
        summary["attack_cat_counts"] = {str(k): int(v) for k, v in data.attack_cat.value_counts().items()}
    return summary


def main() -> int:
    args = parse_args()
    if args.demo:
        from src.demo_data import write_demo_files
        args.data_dir = str(write_demo_files(PROJECT_DIR / "data" / "demo"))
        print("\n" + "!" * 70)
        print("DEMO MODE: training on RANDOMLY GENERATED data, not UNSW-NB15.")
        print("Scores are meaningless. Do NOT use them in your report.")
        print("!" * 70 + "\n")
    data_dir, models_dir, reports_dir = Path(args.data_dir), Path(args.models_dir), Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Find and load the dataset --------------------------------------
    files = find_dataset_files(data_dir)
    if files["train"] is None or files["test"] is None:
        print("\nERROR: Dataset files not found.")
        print(f"Please place these two files in the data folder:\n"
              f"  - {TRAIN_FILE_NAME}\n  - {TEST_FILE_NAME}")
        print("Download them from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("(See README.md, section 'Getting the dataset'.)")
        return 1

    try:
        print("Loading and validating the training file ...")
        train = load_and_validate(files["train"], require_label=True, max_rows=10_000_000, max_mb=2000)
        print("Loading and validating the testing file ...")
        test = load_and_validate(files["test"], require_label=True, max_rows=10_000_000, max_mb=2000)
    except DataValidationError as exc:
        print(f"\nERROR: {exc}")
        return 1

    for name, d in (("training", train), ("testing", test)):
        for w in d.warnings:
            print(f"  [{name}] note: {w}")

    X_train_full, y_train_full = train.features, train.labels
    X_test, y_test = test.features, test.labels

    if args.max_train_rows and args.max_train_rows < len(X_train_full):
        X_train_full, _, y_train_full, _ = train_test_split(
            X_train_full, y_train_full, train_size=args.max_train_rows,
            stratify=y_train_full, random_state=RANDOM_SEED)
        X_train_full, y_train_full = X_train_full.reset_index(drop=True), y_train_full.reset_index(drop=True)
        print(f"Using a random sample of {len(X_train_full):,} training rows (--max-train-rows).")

    print(f"\nTraining rows: {len(X_train_full):,}   Testing rows: {len(X_test):,}")
    print(f"Training class counts (0=normal, 1=attack): {y_train_full.value_counts().to_dict()}")
    print(f"Testing  class counts (0=normal, 1=attack): {y_test.value_counts().to_dict()}")

    # ---- 2. Validation split (for choosing a model only) -------------------
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_SEED)

    candidates = [c for c in CANDIDATES if not (args.skip_random_forest and c == "random_forest")]
    results: dict[str, dict] = {}
    fitted_full: dict = {}

    for name in candidates:
        print(f"\n=== {MODEL_DESCRIPTIONS[name]} ===")
        # ---- 3a. Fit on the training part, score on the validation part
        val_pipe = build_pipeline(name)
        secs = timed_fit(val_pipe, X_fit, y_fit)
        val_metrics = evaluate_pipeline(val_pipe, X_val, y_val)
        print(f"  validation F1 = {val_metrics['f1']:.4f}   (fit time {secs:.1f}s)")

        # ---- 3b. Re-fit on ALL training rows, then score once on the official test set
        final_pipe = build_pipeline(name)
        secs = timed_fit(final_pipe, X_train_full, y_train_full)
        test_metrics = evaluate_pipeline(final_pipe, X_test, y_test)
        print(f"  test accuracy={test_metrics['accuracy']:.4f}  precision={test_metrics['precision']:.4f}  "
              f"recall={test_metrics['recall']:.4f}  f1={test_metrics['f1']:.4f}")

        results[name] = {
            "description": MODEL_DESCRIPTIONS[name],
            "train_seconds": round(secs, 2),
            "validation": {k: v for k, v in val_metrics.items() if k != "roc_curve"},
            "test": test_metrics,
            "feature_importances": get_feature_importances(final_pipe),
        }
        fitted_full[name] = final_pipe

    # ---- 4. Choose the model using VALIDATION F1 only ----------------------
    best = max(results, key=lambda n: results[n]["validation"]["f1"])
    print(f"\nSelected model (best validation F1): {MODEL_DESCRIPTIONS[best]}")

    # ---- 5. Save everything -------------------------------------------------
    metadata = make_metadata(best, train_rows=len(X_train_full),
                             extra={"selection_rule": "highest F1 on a validation split of the training file",
                                    "test_rows": len(X_test), "demo_data": bool(args.demo)})
    save_artifacts(fitted_full[best], metadata, models_dir)

    metrics_doc = {"demo_data": bool(args.demo), "selected_model": best, "selection_rule": metadata["selection_rule"],
                   "random_seed": RANDOM_SEED, "models": results}
    (reports_dir / METRICS_FILE).write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")

    summary_doc = {"training_file": summarise_split("training", train),
                   "testing_file": summarise_split("testing", test)}
    (reports_dir / DATASET_SUMMARY_FILE).write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")

    print(f"\nSaved model to:   {models_dir}")
    print(f"Saved reports to: {reports_dir}")
    print("Next step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
