"""
=============================================================================
STAGE 3: STRICT HOLDOUT TEST EVALUATION (COMPLETELY UNSEEN DATA)
=============================================================================
Author: EPICS DSN3099
Evaluates the generalized, regularized sensor depth engine on the 15% strict holdout:
  - sensor_test.csv (353 samples never seen during training or tuning)
Calculates:
  1. Regression Accuracy: MAE, RMSE, MedAE, R^2, Pearson r, Spearman rho
  2. Delta Accuracies: delta < 1.10, delta < 1.25, delta < 1.56
  3. Errors: AbsRel, SqRel, SILog
  4. Classification Metrics: Accuracy, Balanced Acc, Macro & Weighted F1, Kappa, MCC
  5. Full 3x3 Confusion Matrix & Per-Class Precision/Recall
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef
)

def evaluate_on_unseen_test():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    test_csv = os.path.join(project_root, "pothole_sensor_depth_engine", "data", "sensor_test.csv")
    train_csv = os.path.join(project_root, "pothole_sensor_depth_engine", "data", "sensor_train.csv")
    models_dir = os.path.join(project_root, "pothole_sensor_depth_engine", "models")
    reports_dir = os.path.join(project_root, "pothole_sensor_depth_engine", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("EVALUATING SENSOR DEPTH ENGINE ON STRICT UNSEEN TEST HOLDOUT")
    print("=" * 75)

    test_df = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)
    print(f"Test samples (strictly holdout): {len(test_df)}")

    with open(os.path.join(models_dir, "sensor_depth_regressor.pkl"), "rb") as f:
        regressor = pickle.load(f)
    with open(os.path.join(models_dir, "sensor_tier_classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
    with open(os.path.join(models_dir, "sensor_feature_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "feature_names.json"), "r") as f:
        feat_cols = json.load(f)

    # ── Test Set Predictions ──
    X_test = np.nan_to_num(test_df[feat_cols].values, nan=0.0)
    y_test_depth = test_df["target_depth_cm"].values
    y_test_tier = test_df["severity_tier"].values

    X_test_sc = scaler.transform(X_test)
    y_pred_depth = np.clip(regressor.predict(X_test_sc), 0.5, 15.0)
    y_pred_tier = classifier.predict(X_test_sc)

    # ── Train Set Performance for Overfitting Gap Analysis ──
    X_train = np.nan_to_num(train_df[feat_cols].values, nan=0.0)
    y_tr_depth = train_df["target_depth_cm"].values
    y_tr_tier = train_df["severity_tier"].values
    X_tr_sc = scaler.transform(X_train)
    p_tr_d = regressor.predict(X_tr_sc)
    p_tr_t = classifier.predict(X_tr_sc)

    train_mae = mean_absolute_error(y_tr_depth, p_tr_d)
    train_acc = accuracy_score(y_tr_tier, p_tr_t) * 100.0

    # ── Compute Test Metrics ──
    mae = mean_absolute_error(y_test_depth, y_pred_depth)
    mse = mean_squared_error(y_test_depth, y_pred_depth)
    rmse = np.sqrt(mse)
    median_ae = np.median(np.abs(y_test_depth - y_pred_depth))
    mape = np.mean(np.abs((y_test_depth - y_pred_depth) / y_test_depth)) * 100.0
    r2 = r2_score(y_test_depth, y_pred_depth)

    pearson_r, _ = stats.pearsonr(y_test_depth, y_pred_depth)
    spearman_rho, _ = stats.spearmanr(y_test_depth, y_pred_depth)

    ratio = np.maximum(y_pred_depth / (y_test_depth + 1e-6), y_test_depth / (y_pred_depth + 1e-6))
    delta_110 = float(np.mean(ratio < 1.10) * 100.0)
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    delta_156 = float(np.mean(ratio < (1.25 ** 2)) * 100.0)

    abs_rel = float(np.mean(np.abs(y_pred_depth - y_test_depth) / y_test_depth))
    sq_rel = float(np.mean(((y_pred_depth - y_test_depth) ** 2) / y_test_depth))

    log_diff = np.log(y_pred_depth) - np.log(y_test_depth)
    silog = float(np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)) * 100.0)

    # Classification Metrics
    acc = accuracy_score(y_test_tier, y_pred_tier) * 100.0
    bal_acc = balanced_accuracy_score(y_test_tier, y_pred_tier) * 100.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test_tier, y_pred_tier, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test_tier, y_pred_tier, average="weighted")
    kappa = cohen_kappa_score(y_test_tier, y_pred_tier)
    mcc = matthews_corrcoef(y_test_tier, y_pred_tier)
    cm = confusion_matrix(y_test_tier, y_pred_tier, labels=[1, 2, 3])

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test_tier, y_pred_tier, labels=[1, 2, 3])
    tier_labels = ["Tier 1: Minor", "Tier 2: Moderate", "Tier 3: Critical"]
    per_class_metrics = {}
    for i, tname in enumerate(tier_labels):
        per_class_metrics[tname] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1_score": round(float(f1_per[i]), 4),
            "sample_support": int(sup_per[i])
        }

    # Overfitting Gap Assessment
    mae_gap = abs(train_mae - mae)
    acc_gap = abs(train_acc - acc)
    overfit_status = "HEALTHY GENERALIZATION" if (acc_gap < 8.0 and r2 > 0.85) else "OVERFITTING RISK"

    evaluation_report = {
        "evaluation_protocol": "Strict 15% Holdout Test Split (Never touched during training)",
        "dataset_metadata": {
            "test_samples": len(test_df),
            "train_samples": len(train_df),
            "features_utilized": len(feat_cols),
            "depth_range_cm": {
                "min": round(float(np.min(y_test_depth)), 2),
                "max": round(float(np.max(y_test_depth)), 2),
                "mean": round(float(np.mean(y_test_depth)), 2)
            }
        },
        "overfitting_gap_analysis": {
            "train_mae_cm": round(train_mae, 4),
            "test_mae_cm": round(mae, 4),
            "mae_gap_cm": round(mae_gap, 4),
            "train_tier_accuracy_pct": round(train_acc, 2),
            "test_tier_accuracy_pct": round(acc, 2),
            "accuracy_gap_pct": round(acc_gap, 2),
            "overfitting_verdict": overfit_status
        },
        "regression_depth_metrics": {
            "mean_absolute_error_mae_cm": round(mae, 4),
            "median_absolute_error_cm": round(median_ae, 4),
            "root_mean_squared_error_rmse_cm": round(rmse, 4),
            "mean_absolute_percentage_error_mape_pct": round(mape, 2),
            "coefficient_of_determination_r2": round(r2, 4),
            "pearson_correlation_coefficient_r": round(pearson_r, 4),
            "spearman_rank_correlation_rho": round(spearman_rho, 4)
        },
        "depth_accuracy_threshold_benchmarks": {
            "delta_less_than_1_10_strict_pct": round(delta_110, 2),
            "delta_less_than_1_25_standard_pct": round(delta_125, 2),
            "delta_less_than_1_56_relaxed_pct": round(delta_156, 2)
        },
        "relative_and_scale_invariant_errors": {
            "absolute_relative_error_abs_rel": round(abs_rel, 4),
            "squared_relative_error_sq_rel": round(sq_rel, 4),
            "scale_invariant_logarithmic_error_silog": round(silog, 4)
        },
        "severity_tier_classification_metrics": {
            "overall_accuracy_pct": round(acc, 2),
            "balanced_accuracy_pct": round(bal_acc, 2),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1_score": round(float(f1_macro), 4),
            "weighted_f1_score": round(float(f1_weighted), 4),
            "cohens_kappa_score": round(float(kappa), 4),
            "matthews_correlation_coefficient_mcc": round(float(mcc), 4),
            "confusion_matrix_3x3_rows_gt_cols_pred": cm.tolist(),
            "per_tier_metrics": per_class_metrics
        }
    }

    report_path = os.path.join(reports_dir, "sensor_depth_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=4)

    print(f"\nSaved Test Audit Report -> {report_path}")
    print("-" * 75)
    print("TEST METRICS SUMMARY (STRICT UNSEEN HOLDOUT)")
    print("-" * 75)
    print(f"  * Overfitting Verdict           : {overfit_status}")
    print(f"  * Train Acc vs Test Acc         : {train_acc:.2f}% vs {acc:.2f}% (Gap: {acc_gap:.2f}%)")
    print(f"  * Depth MAE (cm)                : {mae:.4f} cm (Train: {train_mae:.4f} cm)")
    print(f"  * Depth RMSE (cm)               : {rmse:.4f} cm")
    print(f"  * R^2 Determination Score       : {r2:.4f}")
    print(f"  * Pearson Correlation r         : {pearson_r:.4f}")
    print(f"  * Delta < 1.10 Accuracy         : {delta_110:.2f} %")
    print(f"  * Delta < 1.25 Accuracy         : {delta_125:.2f} %")
    print(f"  * AbsRel Error                  : {abs_rel:.4f}")
    print(f"  * SILog Error                   : {silog:.4f}")
    print(f"  * Overall Tier Accuracy         : {acc:.2f} %")
    print(f"  * Macro F1-Score                : {f1_macro:.4f}")
    print(f"  * Cohen's Kappa Score           : {kappa:.4f}")
    print("-" * 75)
    print("Confusion Matrix on Unseen Test (Rows: Ground Truth, Columns: Predicted):")
    print(f"                Pred T1    Pred T2    Pred T3")
    print(f"  True T1       {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10}")
    print(f"  True T2       {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10}")
    print(f"  True T3       {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    evaluate_on_unseen_test()
