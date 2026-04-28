"""
src/pipeline.py
Master orchestration script that runs the full ML pipeline end-to-end:

  1. Loads the raw Nakaseke Hospital dataset (RIF_final.dta)
  2. Cleans and validates every column
  3. Engineers clinical composite features
  4. Generates and saves all EDA visualisations
  5. Trains and tunes six ML models (tracked in MLflow)
  6. Evaluates the winning model on the held-out test set
  7. Computes SHAP feature importance values
  8. Saves the final model artefacts to disk

Usage:
    python -m src.pipeline
    python train.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


def run() -> dict:
    """Execute the full ML pipeline and return the results dict."""

    from src.data.loader   import load_rif
    from src.data.cleaner  import clean_rif
    from src.data.features import engineer_features, get_feature_names

    # Step 1: Load
    logger.info("--- STEP 1/5  Load raw data ---")
    rif_raw = load_rif("data/raw/RIF_final.dta")

    # Step 2: Clean
    logger.info("--- STEP 2/5  Clean & validate ---")
    clean = clean_rif(rif_raw)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    clean.to_csv("data/processed/nakaseke_clean.csv", index=False)
    logger.info("Cleaned dataset saved -> data/processed/nakaseke_clean.csv")

    # Step 3: Feature engineering
    logger.info("--- STEP 3/5  Feature engineering ---")
    df = engineer_features(clean)
    feature_names = get_feature_names(df, target="hypertension")
    logger.info("Final feature set: %d predictors x %d rows", len(feature_names), len(df))

    # Step 4: EDA plots
    logger.info("--- STEP 4/5  EDA visualisations ---")
    from src.visualization.plots import run_eda_plots
    run_eda_plots(df, target="hypertension", output_dir=Path("models/figures"))

    # Step 5: Train & evaluate
    logger.info("--- STEP 5/5  Model training ---")
    from src.models.trainer import train_all
    from src.visualization.plots import (
        plot_roc_curves, plot_pr_curves,
        plot_confusion_matrix, plot_shap_summary,
    )

    results = train_all(df, feature_names)

    # Evaluation plots
    roc_data = {n: {"fpr": m["fpr"], "tpr": m["tpr"], "auc": m["auc"]}
                for n, m in results["all_metrics"].items()}
    plot_roc_curves(roc_data)

    pr_data = {n: {"precision": m["precision"], "recall": m["recall"],
                    "avg_precision": m["avg_precision"]}
               for n, m in results["all_metrics"].items()}
    plot_pr_curves(pr_data)

    plot_confusion_matrix(
        results["test_metrics"]["confusion_matrix"],
        model_name=results["best_name"],
    )

    if results["shap_values"] is not None:
        plot_shap_summary(results["shap_values"], results["feature_names"])

    # Summary
    m = results["test_metrics"]
    logger.info(
        "\n"
        "=================================================\n"
        "         FINAL TEST-SET PERFORMANCE              \n"
        "=================================================\n"
        "  Model      : %s\n"
        "  ROC-AUC    : %.4f\n"
        "  Avg Prec   : %.4f\n"
        "  F1 Score   : %.4f\n"
        "  Recall     : %.4f\n"
        "  Precision  : %.4f\n"
        "  Accuracy   : %.4f\n"
        "=================================================",
        results["best_name"],
        m["roc_auc"], m["avg_precision"],
        m["f1"], m["recall"], m["precision"], m["accuracy"],
    )

    return results


if __name__ == "__main__":
    run()
