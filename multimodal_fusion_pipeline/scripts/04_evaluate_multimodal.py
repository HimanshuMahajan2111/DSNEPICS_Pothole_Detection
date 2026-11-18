"""
=============================================================================
STAGE 4: COMPREHENSIVE MULTIMODAL EVALUATION (STRICT HOLDOUT TEST)
=============================================================================
Author: EPICS DSN3099
Evaluates the unified Image + Color Segmentation + Sensor Depth Model on:
  - 427 unseen test samples (strict holdout)
  - Regression: MAE, RMSE, MedAE, MAPE, R^2, Pearson r, Spearman rho
  - Standard Depth Accuracies: delta < 1.10, delta < 1.25, delta < 1.56
  - Relative Error: AbsRel, SqRel, SILog
  - Severity Tier Classification: Accuracy, Balanced Acc, F1, Kappa, MCC
  - 3x3 Confusion Matrix & Per-tier Precision/Recall
=============================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef
)

def evaluate_multimodal_test():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "multimodal_fusion_pipeline", "data")
    models_dir = os.path.join(project_root, "multimodal_fusion_pipeline", "models")
    reports_dir = os.path.join(project_root, "multimodal_fusion_pipeline", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("EVALUATING MULTIMODAL FUSION ENGINE ON STRICT UNSEEN TEST SPLIT")
    print("=" * 75)

    test_df = pd.read_csv(os.path.join(data_dir, "fusion_test.csv"))
    train_df = pd.read_csv(os.path.join(data_dir, "fusion_train.csv"))

    with open(os.path.join(models_dir, "fusion_depth_regressor.pkl"), "rb") as f:
        reg = pickle.load(f)
    with open(os.path.join(models_dir, "fusion_tier_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(models_dir, "fusion_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(models_dir, "feature_names.json"), "r") as f:
        feat_cols = json.load(f)

    # Predictions on unseen test set
    X_test = np.nan_to_num(test_df[feat_cols].values, nan=0.0)
    y_test_d = test_df["target_depth_cm"].values
    y_test_t = test_df["severity_tier"].values

    X_test_sc = scaler.transform(X_test)
    p_test_d = np.clip(reg.predict(X_test_sc), 0.5, 15.0)
    p_test_t = clf.predict(X_test_sc)

    # Train predictions for gap analysis
    X_tr = np.nan_to_num(train_df[feat_cols].values, nan=0.0)
    y_tr_d = train_df["target_depth_cm"].values
    y_tr_t = train_df["severity_tier"].values
    X_tr_sc = scaler.transform(X_tr)
    p_tr_d = reg.predict(X_tr_sc)
    p_tr_t = clf.predict(X_tr_sc)

    tr_mae = mean_absolute_error(y_tr_d, p_tr_d)
    tr_acc = accuracy_score(y_tr_t, p_tr_t) * 100.0

    # Test Metrics
    mae = mean_absolute_error(y_test_d, p_test_d)
    mse = mean_squared_error(y_test_d, p_test_d)
    rmse = np.sqrt(mse)
    median_ae = np.median(np.abs(y_test_d - p_test_d))
    mape = np.mean(np.abs((y_test_d - p_test_d) / y_test_d)) * 100.0
    r2 = r2_score(y_test_d, p_test_d)

    pearson_r, _ = stats.pearsonr(y_test_d, p_test_d)
    spearman_rho, _ = stats.spearmanr(y_test_d, p_test_d)

    ratio = np.maximum(p_test_d / (y_test_d + 1e-6), y_test_d / (p_test_d + 1e-6))
    delta_110 = float(np.mean(ratio < 1.10) * 100.0)
    delta_125 = float(np.mean(ratio < 1.25) * 100.0)
    delta_156 = float(np.mean(ratio < (1.25 ** 2)) * 100.0)

    abs_rel = float(np.mean(np.abs(p_test_d - y_test_d) / y_test_d))
    sq_rel = float(np.mean(((p_test_d - y_test_d) ** 2) / y_test_d))

    log_diff = np.log(p_test_d) - np.log(y_test_d)
    silog = float(np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2)) * 100.0)

    # Classification
    acc = accuracy_score(y_test_t, p_test_t) * 100.0
    bal_acc = balanced_accuracy_score(y_test_t, p_test_t) * 100.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test_t, p_test_t, average="macro")
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test_t, p_test_t, average="weighted")
    kappa = cohen_kappa_score(y_test_t, p_test_t)
    mcc = matthews_corrcoef(y_test_t, p_test_t)
    cm = confusion_matrix(y_test_t, p_test_t, labels=[1, 2, 3])

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test_t, p_test_t, labels=[1, 2, 3])
    tier_names = ["Tier 1: Minor", "Tier 2: Moderate", "Tier 3: Critical"]
    per_class = {}
    for i, tname in enumerate(tier_names):
        per_class[tname] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1_score": round(float(f1_per[i]), 4),
            "support": int(sup_per[i])
        }

    report = {
        "architecture": "Multimodal Tri-Branch Fusion (Vision + Color Segmentation + Quarter-Car IMU)",
        "test_samples": len(test_df),
        "train_samples": len(train_df),
        "overfitting_gap": {
            "train_mae_cm": round(tr_mae, 4),
            "test_mae_cm": round(mae, 4),
            "train_tier_acc_pct": round(tr_acc, 2),
            "test_tier_acc_pct": round(acc, 2),
            "verdict": "HEALTHY GENERALIZATION" if abs(tr_acc - acc) < 6.0 else "OVERFITTING RISK"
        },
        "depth_regression_metrics": {
            "mae_cm": round(mae, 4),
            "median_ae_cm": round(median_ae, 4),
            "rmse_cm": round(rmse, 4),
            "mape_pct": round(mape, 2),
            "r2_score": round(r2, 4),
            "pearson_r": round(pearson_r, 4),
            "spearman_rho": round(spearman_rho, 4)
        },
        "depth_accuracy_thresholds": {
            "delta_1_10_acc_pct": round(delta_110, 2),
            "delta_1_25_acc_pct": round(delta_125, 2),
            "delta_1_56_acc_pct": round(delta_156, 2),
            "abs_rel_error": round(abs_rel, 4),
            "sq_rel_error": round(sq_rel, 4),
            "silog_error": round(silog, 4)
        },
        "severity_tier_classification": {
            "accuracy_pct": round(acc, 2),
            "balanced_accuracy_pct": round(bal_acc, 2),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "cohens_kappa": round(float(kappa), 4),
            "matthews_corrcoef_mcc": round(float(mcc), 4),
            "confusion_matrix_3x3": cm.tolist(),
            "per_tier_metrics": per_class
        }
    }

    report_p = os.path.join(reports_dir, "multimodal_fusion_evaluation_report.json")
    with open(report_p, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nSaved Multimodal Audit Report -> {report_p}")
    print("-" * 75)
    print("MULTIMODAL FUSION TEST METRICS SUMMARY")
    print("-" * 75)
    print(f"  * Train vs Test Severity Accuracy: {tr_acc:.2f}% vs {acc:.2f}%")
    print(f"  * Depth MAE (cm)                : {mae:.4f} cm (RMSE: {rmse:.4f} cm)")
    print(f"  * Determination Score (R^2)     : {r2:.4f}")
    print(f"  * Pearson Correlation r         : {pearson_r:.4f}")
    print(f"  * Delta < 1.10 Accuracy         : {delta_110:.2f} %")
    print(f"  * Delta < 1.25 Accuracy         : {delta_125:.2f} %")
    print(f"  * SILog Logarithmic Error       : {silog:.4f}")
    print(f"  * Overall Severity Tier Acc     : {acc:.2f} %")
    print(f"  * Balanced Tier Accuracy        : {bal_acc:.2f} %")
    print(f"  * Macro F1-Score                : {f1_macro:.4f}")
    print(f"  * Cohen's Kappa                 : {kappa:.4f}")
    print("-" * 75)
    print("Test Confusion Matrix (Rows: Ground Truth, Columns: Predicted):")
    print(f"                Pred T1    Pred T2    Pred T3")
    print(f"  True T1       {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10}")
    print(f"  True T2       {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10}")
    print(f"  True T3       {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    evaluate_multimodal_test()
