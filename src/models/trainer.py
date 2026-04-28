"""
src/models/trainer.py
_____________________
End-to-end model training with:
  _ Train / validation / test split (stratified, reproducible)
  _ SMOTE oversampling on the training set only
  _ Standard scaling for linear models
  _ Six candidate models benchmarked on the same CV folds
  _ Optuna hyperparameter optimisation for the top two contenders
  _ Full MLflow experiment tracking (params, metrics, artefacts)
  _ SHAP values computed and saved for the winning model

Design choices
______________
All data-leaking steps (scaler fit, SMOTE) happen INSIDE the training fold;
the test set is never touched until the final evaluation so held-out metrics
are unbiased estimates of real-world performance.
"""

import time
import logging
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import CFG

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

TARGET = CFG.target.column
SEED   = CFG.training.random_state


# _____________________________________________________________________________
# Data splitting
# _____________________________________________________________________________

def split_data(df: pd.DataFrame, feature_names: list[str]) -> tuple:
    """
    Stratified three-way split:  train / validation / held-out test.

    Stratification on the target ensures the positive/negative ratio is
    preserved across all splits, which matters especially for the validation
    and test sets where small imbalances could mislead threshold selection.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[feature_names].astype(float)
    y = df[TARGET].astype(int)

    # First carve off the test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=CFG.training.test_size,
        stratify=y,
        random_state=SEED,
    )
    # Then split the remainder into train / val
    val_fraction = CFG.training.val_size / (1 - CFG.training.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_fraction,
        stratify=y_tv,
        random_state=SEED,
    )

    logger.info(
        "Split: train=%d  val=%d  test=%d  (pos rate: train=%.2f val=%.2f test=%.2f)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# _____________________________________________________________________________
# Candidate model definitions
# _____________________________________________________________________________

_IMPUTER = lambda: SimpleImputer(strategy="median")
_SMOTE   = lambda: SMOTE(k_neighbors=CFG.training.smote_k_neighbors, random_state=SEED)


def _build_lr_pipeline():
    """Logistic Regression inside an impute -> SMOTE -> scale pipeline."""
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def _build_rf_pipeline():
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("clf",     RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=SEED, n_jobs=-1)),
    ])


def _build_gb_pipeline():
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("clf",     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, random_state=SEED)),
    ])


def _build_xgb_pipeline():
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("clf",     XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                   eval_metric="logloss",
                                   scale_pos_weight=1, random_state=SEED,
                                   verbosity=0, n_jobs=-1)),
    ])


def _build_lgbm_pipeline():
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("clf",     LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                    class_weight="balanced", random_state=SEED,
                                    verbose=-1, n_jobs=-1)),
    ])


def _build_knn_pipeline():
    return ImbPipeline([
        ("imputer", _IMPUTER()),
        ("smote",   _SMOTE()),
        ("scaler",  StandardScaler()),
        ("clf",     KNeighborsClassifier(n_neighbors=11, n_jobs=-1)),
    ])


CANDIDATE_BUILDERS = {
    "Logistic Regression":     _build_lr_pipeline,
    "Random Forest":           _build_rf_pipeline,
    "Gradient Boosting":       _build_gb_pipeline,
    "XGBoost":                 _build_xgb_pipeline,
    "LightGBM":                _build_lgbm_pipeline,
    "K-Nearest Neighbours":    _build_knn_pipeline,
}


# _____________________________________________________________________________
# Evaluation helper
# _____________________________________________________________________________

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Return a rich metrics dict for a fitted model against the test set.
    Includes ROC / PR curve arrays so they can be plotted later.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    fpr, tpr, _  = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "roc_auc":       roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "accuracy":      report["accuracy"],
        "f1":            report["1"]["f1-score"],
        "precision":     report["1"]["precision"],
        "recall":        report["1"]["recall"],
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr, "tpr": tpr,
        "prec_curve": prec, "rec_curve": rec,
        "y_pred": y_pred, "y_proba": y_proba,
    }


# _____________________________________________________________________________
# Optuna hyperparameter search for XGBoost (best performing tree model)
# _____________________________________________________________________________

def _optuna_xgb(X_train, y_train) -> dict:
    """
    Bayesian hyperparameter optimisation using Optuna's TPE sampler.
    Optimises 5-fold stratified CV AUC-ROC on the training set.
    """
    cv = StratifiedKFold(n_splits=CFG.training.cv_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",  100, 600),
            "max_depth":         trial.suggest_int("max_depth",      3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample",    0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5),
        }
        pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote",   SMOTE(k_neighbors=5, random_state=SEED)),
            ("clf",     XGBClassifier(**params, eval_metric="logloss", random_state=SEED,
                                      verbosity=0, n_jobs=-1)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=CFG.optuna.n_trials,
                   timeout=CFG.optuna.timeout, show_progress_bar=False)
    return study.best_params


def _optuna_lgbm(X_train, y_train) -> dict:
    """Same optimisation loop for LightGBM."""
    cv = StratifiedKFold(n_splits=CFG.training.cv_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators":   trial.suggest_int("n_estimators",  100, 600),
            "max_depth":      trial.suggest_int("max_depth",      3, 9),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":     trial.suggest_int("num_leaves",     20, 150),
            "subsample":      trial.suggest_float("subsample",    0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":      trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda":     trial.suggest_float("reg_lambda", 0, 2),
        }
        pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote",   SMOTE(k_neighbors=5, random_state=SEED)),
            ("clf",     LGBMClassifier(**params, class_weight="balanced",
                                       random_state=SEED, verbose=-1, n_jobs=-1)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=CFG.optuna.n_trials,
                   timeout=CFG.optuna.timeout, show_progress_bar=False)
    return study.best_params


# _____________________________________________________________________________
# Main training orchestration
# _____________________________________________________________________________

def train_all(df: pd.DataFrame, feature_names: list[str]) -> dict:
    """
    Full training pipeline:
      1. Split data
      2. Benchmark six candidate models (default hyper-params + CV)
      3. Select top-2 by validation AUC
      4. Run Optuna for each top model
      5. Retrain best on train+val, evaluate on held-out test
      6. Compute SHAP values for the winner
      7. Log everything to MLflow
      8. Persist the winning model to disk

    Parameters
    ----------
    df            : Clean, feature-engineered DataFrame
    feature_names : Ordered list of predictor column names

    Returns
    -------
    dict  _ {"best_model": pipeline, "best_name": str,
             "test_metrics": dict, "shap_values": np.ndarray,
             "all_metrics": {model_name: dict}}
    """
    mlflow.set_tracking_uri(CFG.mlflow.tracking_uri)
    mlflow.set_experiment(CFG.mlflow.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, feature_names)

    # __ Step 1: benchmark ____________________________________________________
    cv = StratifiedKFold(n_splits=CFG.training.cv_folds, shuffle=True, random_state=SEED)
    benchmark = {}

    logger.info("___ Benchmarking %d candidate models ___", len(CANDIDATE_BUILDERS))
    with mlflow.start_run(run_name="benchmark"):
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val",   len(X_val))
        mlflow.log_param("n_test",  len(X_test))
        mlflow.log_param("features", feature_names)

        for name, builder in CANDIDATE_BUILDERS.items():
            t0 = time.time()
            pipe = builder()
            cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv,
                                       scoring="roc_auc", n_jobs=-1)
            # Fit on full train, evaluate on val
            pipe.fit(X_train, y_train)
            val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
            elapsed = time.time() - t0

            benchmark[name] = {
                "cv_auc_mean":  cv_aucs.mean(),
                "cv_auc_std":   cv_aucs.std(),
                "val_auc":      val_auc,
                "fit_time_s":   elapsed,
                "model":        pipe,
            }
            mlflow.log_metrics({
                f"{name}_cv_auc":  cv_aucs.mean(),
                f"{name}_val_auc": val_auc,
            })
            logger.info("%-25s  CV AUC=%.4f_%.4f  Val AUC=%.4f  (%.1fs)",
                        name, cv_aucs.mean(), cv_aucs.std(), val_auc, elapsed)

    # __ Step 2: pick top-2 by validation AUC _________________________________
    ranked = sorted(benchmark.items(), key=lambda x: x[1]["val_auc"], reverse=True)
    top2   = [ranked[0][0], ranked[1][0]]
    logger.info("Top-2 models for Optuna: %s", top2)

    # __ Step 3: Optuna tuning _________________________________________________
    tuned_pipes = {}
    for name in top2:
        logger.info("Running Optuna for %s _", name)
        with mlflow.start_run(run_name=f"optuna_{name.replace(' ', '_')}"):
            if "XGBoost" in name:
                best_params = _optuna_xgb(X_train, y_train)
                tuned_pipe  = ImbPipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote",   SMOTE(k_neighbors=5, random_state=SEED)),
                    ("clf",     XGBClassifier(**best_params, eval_metric="logloss",
                                              random_state=SEED, verbosity=0, n_jobs=-1)),
                ])
            elif "LightGBM" in name:
                best_params = _optuna_lgbm(X_train, y_train)
                tuned_pipe  = ImbPipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("smote",   SMOTE(k_neighbors=5, random_state=SEED)),
                    ("clf",     LGBMClassifier(**best_params, class_weight="balanced",
                                               random_state=SEED, verbose=-1, n_jobs=-1)),
                ])
            else:
                # For non-tree models just re-use the benchmark model
                best_params = {}
                tuned_pipe  = benchmark[name]["model"]

            tuned_pipe.fit(X_train, y_train)
            val_auc = roc_auc_score(y_val, tuned_pipe.predict_proba(X_val)[:, 1])
            tuned_pipes[name] = {"pipe": tuned_pipe, "val_auc": val_auc}

            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("tuned_val_auc", val_auc)
            logger.info("  %s tuned val AUC: %.4f", name, val_auc)

    # __ Step 4: Pick winner, retrain on train+val _____________________________
    best_name = max(tuned_pipes, key=lambda k: tuned_pipes[k]["val_auc"])
    best_pipe = tuned_pipes[best_name]["pipe"]
    logger.info("Winner: %s", best_name)

    # Combine train + val for final fit
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    best_pipe.fit(X_trainval, y_trainval)

    # __ Step 5: Test-set evaluation ___________________________________________
    test_metrics = evaluate_model(best_pipe, X_test, y_test)

    with mlflow.start_run(run_name=f"final_{best_name.replace(' ', '_')}"):
        mlflow.log_param("model_name",  best_name)
        mlflow.log_param("n_features",  len(feature_names))
        mlflow.log_metrics({
            "test_roc_auc":       test_metrics["roc_auc"],
            "test_avg_precision": test_metrics["avg_precision"],
            "test_f1":            test_metrics["f1"],
            "test_recall":        test_metrics["recall"],
            "test_precision":     test_metrics["precision"],
            "test_accuracy":      test_metrics["accuracy"],
        })
        mlflow.sklearn.log_model(best_pipe, "model")

    # __ Step 6: SHAP values ___________________________________________________
    shap_values = _compute_shap(best_pipe, X_test, feature_names)

    # __ Step 7: Persist artefacts _____________________________________________
    model_dir = Path(CFG.paths.model_dir)
    model_dir.mkdir(exist_ok=True)

    model_path = Path(CFG.paths.best_model)
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    logger.info("Model saved _ %s", model_path)

    np.save(Path(CFG.paths.shap_values), shap_values)
    pd.Series(feature_names).to_csv(model_dir / "feature_names.csv", index=False)

    # Collect ROC/PR data for all benchmark models for plotting
    all_metrics = {}
    for name, info in benchmark.items():
        m = evaluate_model(info["model"], X_test, y_test)
        all_metrics[name] = {
            "fpr":           m["fpr"],
            "tpr":           m["tpr"],
            "auc":           m["roc_auc"],
            "precision":     m["prec_curve"],
            "recall":        m["rec_curve"],
            "avg_precision": m["avg_precision"],
        }

    logger.info(
        "Training complete _ Best: %s _ Test AUC: %.4f _ F1: %.4f",
        best_name, test_metrics["roc_auc"], test_metrics["f1"],
    )

    return {
        "best_model":   best_pipe,
        "best_name":    best_name,
        "test_metrics": test_metrics,
        "shap_values":  shap_values,
        "all_metrics":  all_metrics,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
    }


# _____________________________________________________________________________
# SHAP computation
# _____________________________________________________________________________

def _compute_shap(pipe, X_test: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Compute SHAP values using the TreeExplainer (fast, exact for tree models).
    Falls back to KernelExplainer for non-tree models.
    """
    try:
        clf = pipe.named_steps["clf"]
        X_test_np = X_test.values.astype(float)
        explainer  = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test_np)
        # Binary classification: TreeExplainer may return list [neg, pos]
        if isinstance(sv, list):
            sv = sv[1]
        return sv
    except Exception as e:
        logger.warning("TreeExplainer failed (%s); falling back to KernelExplainer _", e)
        pred_fn    = lambda x: pipe.predict_proba(pd.DataFrame(x, columns=feature_names))[:, 1]
        background = shap.sample(X_test, 50)
        explainer  = shap.KernelExplainer(pred_fn, background)
        return explainer.shap_values(X_test.values[:100])
