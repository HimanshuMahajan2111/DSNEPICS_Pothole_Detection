"""
=============================================================================
STEP 3: COMPREHENSIVE EVALUATION (V2 — Honest Metrics)
=============================================================================
Evaluates the trained MiDaS-calibrated depth model with ALL standard metrics:
  - Depth: MAE, RMSE, AbsRel, SqRel, SILog, δ<1.25/1.56/1.95
  - Dimensions: Width MAE, Length MAE, Volume relative error
  - Classification: Tier accuracy, precision, recall, F1, confusion matrix
  - CV errors: Mean error per fold, std across folds, confidence intervals
  - Overfitting check: Train vs CV gap analysis

Uses the saved cross-validation predictions for HONEST evaluation.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def compute_all_depth_metrics(y_true, y_pred):
    """Compute comprehensive depth estimation metrics."""
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0.5, 15.0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    abs_rel = float(np.mean(np.abs(y_pred - y_true) / (y_true + 1e-8)))
    sq_rel = float(np.mean((y_pred - y_true) ** 2 / (y_true + 1e-8)))

    # Delta threshold accuracies
    ratio = np.maximum(y_pred / (y_true + 1e-8), y_true / (y_pred + 1e-8))
    delta_125 = float(np.mean(ratio < 1.25) * 100)
    delta_156 = float(np.mean(ratio < 1.25 ** 2) * 100)
    delta_195 = float(np.mean(ratio < 1.25 ** 3) * 100)

    # SILog (Scale-Invariant Logarithmic error)
    log_diff = np.log(y_pred + 1e-8) - np.log(y_true + 1e-8)
    silog = float(np.sqrt(np.mean(log_diff ** 2) - np.mean(log_diff) ** 2) * 100)

    # Median Absolute Error (more robust)
    median_ae = float(np.median(np.abs(y_pred - y_true)))

    # Mean Relative Error
    mre = float(np.mean(np.abs(y_pred - y_true) / (y_true + 1e-8)) * 100)

    # Percentage within tolerance bands
    pct_within_05cm = float(np.mean(np.abs(y_pred - y_true) < 0.5) * 100)
    pct_within_1cm = float(np.mean(np.abs(y_pred - y_true) < 1.0) * 100)
    pct_within_2cm = float(np.mean(np.abs(y_pred - y_true) < 2.0) * 100)

    return {
        "mae_cm": round(mae, 4),
        "median_ae_cm": round(median_ae, 4),
        "rmse_cm": round(rmse, 4),
        "mse_cm2": round(mse, 4),
        "r2_score": round(r2, 4),
        "abs_rel_error": round(abs_rel, 4),
        "sq_rel_error": round(sq_rel, 4),
        "silog": round(silog, 2),
        "mean_relative_error_pct": round(mre, 2),
        "delta_lt_1_25_pct": round(delta_125, 2),
        "delta_lt_1_56_pct": round(delta_156, 2),
        "delta_lt_1_95_pct": round(delta_195, 2),
        "pct_within_0_5cm": round(pct_within_05cm, 2),
        "pct_within_1_0cm": round(pct_within_1cm, 2),
        "pct_within_2_0cm": round(pct_within_2cm, 2),
    }


def compute_classification_metrics(y_true, y_pred):
    """Compute comprehensive tier classification metrics."""
    acc = accuracy_score(y_true, y_pred) * 100
    present = sorted(set(y_true) | set(y_pred))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])

    per_class = {}
    tier_names = {1: "Minor", 2: "Moderate", 3: "Critical"}
    for i, label in enumerate(present):
        per_class[f"Tier_{label}_{tier_names.get(label, 'Unknown')}"] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy_pct": round(acc, 2),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix_rows_gt_cols_pred": cm.tolist(),
    }


def overfitting_check(fold_metrics):
    """
    Check for signs of overfitting by analysing fold-level variance.
    High variance across folds = overfitting risk.
    """
    maes = [f["mae_cm"] for f in fold_metrics]
    deltas = [f["delta_125_pct"] for f in fold_metrics]
    tier_accs = [f["tier_accuracy_pct"] for f in fold_metrics]

    mae_mean, mae_std = np.mean(maes), np.std(maes)
    delta_mean, delta_std = np.mean(deltas), np.std(deltas)
    tier_mean, tier_std = np.mean(tier_accs), np.std(tier_accs)

    # Coefficient of variation — high CV suggests instability
    mae_cv = mae_std / (mae_mean + 1e-8) * 100
    delta_cv = delta_std / (delta_mean + 1e-8) * 100
    tier_cv = tier_std / (tier_mean + 1e-8) * 100

    # 95% confidence interval for MAE
    t_crit = stats.t.ppf(0.975, df=len(maes) - 1)
    mae_ci = t_crit * mae_std / np.sqrt(len(maes))

    overfitting_risk = "LOW"
    if mae_cv > 20 or delta_cv > 15 or tier_cv > 10:
        overfitting_risk = "MODERATE"
    if mae_cv > 40 or delta_cv > 30 or tier_cv > 20:
        overfitting_risk = "HIGH"

    return {
        "overfitting_risk_level": overfitting_risk,
        "fold_mae_mean_cm": round(mae_mean, 4),
        "fold_mae_std_cm": round(mae_std, 4),
        "fold_mae_cv_pct": round(mae_cv, 2),
        "fold_mae_95_ci": f"±{mae_ci:.4f} cm",
        "fold_delta_125_mean_pct": round(delta_mean, 2),
        "fold_delta_125_std_pct": round(delta_std, 2),
        "fold_tier_acc_mean_pct": round(tier_mean, 2),
        "fold_tier_acc_std_pct": round(tier_std, 2),
        "interpretation": (
            f"MAE varies by CV={mae_cv:.1f}% across folds. "
            f"{'Stable results suggest good generalisation.' if mae_cv < 20 else 'High variance suggests possible overfitting.'}"
        ),
    }


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    report_dir = os.path.join(project_root, "pothole_depth_pipeline_v2", "reports")
    os.makedirs(report_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 3: COMPREHENSIVE EVALUATION (V2)")
    print("=" * 70)

    # Load training report (contains CV predictions and fold metrics)
    train_report_path = os.path.join(report_dir, "depth_training_report_v2.json")
    if not os.path.exists(train_report_path):
        print(f"ERROR: Run Step 2 first. Missing: {train_report_path}")
        sys.exit(1)

    with open(train_report_path) as f:
        train_report = json.load(f)

    # Load data for independent analysis
    data_csv = os.path.join(project_root, "pothole_depth_pipeline_v2", "data",
                            "midas_depth_features.csv")
    df = pd.read_csv(data_csv)

    # Load models for train-set prediction (overfitting check)
    ckpt_dir = os.path.join(project_root, "pothole_depth_pipeline_v2", "checkpoints")
    with open(os.path.join(ckpt_dir, "final_depth_model_v2.pkl"), "rb") as f:
        final_depth = pickle.load(f)
    with open(os.path.join(ckpt_dir, "final_tier_model_v2.pkl"), "rb") as f:
        final_tier = pickle.load(f)
    with open(os.path.join(ckpt_dir, "final_scaler_v2.pkl"), "rb") as f:
        final_scaler = pickle.load(f)
    with open(os.path.join(ckpt_dir, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    # Add derived features
    from importlib.util import spec_from_file_location, module_from_spec
    train_script = os.path.join(os.path.dirname(__file__), "02_train_calibration.py")
    spec = spec_from_file_location("train_mod", train_script)
    train_mod = module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    df = train_mod.add_derived_features(df)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y_depth = df["ref_depth_cm"].values
    y_tier = df["severity_tier"].values

    # Predict on training set (for overfitting gap analysis)
    X_sc = final_scaler.transform(X)
    train_depth_pred = np.clip(final_depth.predict(X_sc), 0.5, 15.0)
    train_tier_pred = final_tier.predict(X_sc)

    train_mae = mean_absolute_error(y_depth, train_depth_pred)
    train_tier_acc = accuracy_score(y_tier, train_tier_pred) * 100

    cv_mae = train_report["overall_cv_metrics"]["depth_mae_cm"]
    cv_tier = train_report["overall_cv_metrics"]["tier_classification_accuracy_pct"]

    print(f"\n  Overfitting Gap Analysis:")
    print(f"    Train MAE: {train_mae:.4f} cm  |  CV MAE: {cv_mae:.4f} cm  "
          f"|  Gap: {abs(train_mae - cv_mae):.4f} cm")
    print(f"    Train Tier Acc: {train_tier_acc:.2f}%  |  CV Tier Acc: {cv_tier:.2f}%  "
          f"|  Gap: {abs(train_tier_acc - cv_tier):.2f}%")

    # Overfitting check
    overfit_analysis = overfitting_check(train_report["per_fold_metrics"])
    overfit_analysis["train_vs_cv_mae_gap_cm"] = round(abs(train_mae - cv_mae), 4)
    overfit_analysis["train_vs_cv_tier_gap_pct"] = round(abs(train_tier_acc - cv_tier), 2)

    # Compute depth metrics from CV (already in report, but re-derive for completeness)
    cv_metrics = train_report["overall_cv_metrics"]

    # Dimension error analysis (using geometric width/length)
    geo_width = df["phys_width_cm_geo"].values
    ref_width = df["ref_width_cm"].values
    geo_length = df["phys_length_cm_geo"].values
    ref_length = df["ref_length_cm"].values

    width_mae = mean_absolute_error(ref_width, geo_width)
    length_mae = mean_absolute_error(ref_length, geo_length)
    width_mape = float(np.mean(np.abs(geo_width - ref_width) / (ref_width + 1e-8)) * 100)
    length_mape = float(np.mean(np.abs(geo_length - ref_length) / (ref_length + 1e-8)) * 100)

    dim_metrics = {
        "width_mae_cm": round(width_mae, 2),
        "width_mape_pct": round(width_mape, 2),
        "length_mae_cm": round(length_mae, 2),
        "length_mape_pct": round(length_mape, 2),
        "note": "Width/length from frustum projection are identical to reference "
                "(same formula), so these only measure geometric self-consistency.",
    }

    # Classification metrics
    class_metrics = compute_classification_metrics(y_tier, train_tier_pred)
    class_metrics_note = ("These are on the TRAINING set for the final model. "
                         "CV accuracy of {:.2f}% is the honest generalisation estimate.").format(cv_tier)

    # ── Build final report ──
    final_report = {
        "model_description": {
            "architecture": "MiDaS DPT-Hybrid → GradientBoostedRegressor calibration head",
            "depth_backbone": "Intel MiDaS DPT-Hybrid (trained on 12 real RGB-D datasets)",
            "calibration_model": "GradientBoostingRegressor (500 trees, max_depth=5)",
            "classifier": "GradientBoostingClassifier (300 trees, class-weight balanced)",
            "training_method": "5-fold stratified cross-validation, 2× noise augmentation",
        },
        "dataset_statistics": {
            "total_samples": len(df),
            "tier_distribution": {
                f"Tier_{t}": int(c)
                for t, c in zip(*np.unique(y_tier, return_counts=True))
            },
            "depth_range_cm": f"{y_depth.min():.2f} – {y_depth.max():.2f}",
            "depth_mean_cm": round(float(y_depth.mean()), 2),
        },
        "cross_validated_depth_metrics": cv_metrics,
        "cross_validated_tier_classification": {
            "accuracy_pct": cv_tier,
            "note": "These are out-of-fold predictions — the model never trained on these samples.",
        },
        "train_set_tier_classification": class_metrics,
        "dimension_metrics": dim_metrics,
        "overfitting_analysis": overfit_analysis,
        "feature_importances": train_report.get("top_10_feature_importances", []),
        "transparency_notes": [
            "Reference depth labels are FORMULA-DERIVED from bbox geometry + "
            "luminance. No real-world ruler/LiDAR ground truth is available.",
            "MiDaS provides real relative depth trained on 12 RGB-D datasets "
            "(MegaDepth, KITTI, etc.), grounding estimates in real visual depth.",
            "δ<1.25 accuracy measures agreement between MiDaS-calibrated predictions "
            "and formula-derived references, NOT absolute real-world accuracy.",
            "The most deployment-reliable metric is SEVERITY TIER classification — "
            "it determines maintenance action regardless of absolute cm accuracy.",
            "To validate absolute depth, compare against 3-5 potholes measured with "
            "a ruler or depth gauge in the field.",
        ],
    }

    report_path = os.path.join(report_dir, "depth_evaluation_report_v2.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)

    print(f"\n  Full evaluation report → {report_path}")

    # Pretty print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (V2 — MiDaS-backed)")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'CV Depth MAE':<35} {cv_metrics['depth_mae_cm']:>8.3f} cm")
    print(f"  {'CV Depth RMSE':<35} {cv_metrics['depth_rmse_cm']:>8.3f} cm")
    print(f"  {'CV R²':<35} {cv_metrics['depth_r2']:>10.4f}")
    print(f"  {'CV AbsRel':<35} {cv_metrics['abs_rel_error']:>10.4f}")
    print(f"  {'CV SILog':<35} {cv_metrics['silog']:>10.2f}")
    print(f"  {'CV δ<1.25 Accuracy':<35} {cv_metrics['delta_lt_1_25_pct']:>8.2f} %")
    print(f"  {'CV δ<1.25² Accuracy':<35} {cv_metrics['delta_lt_1_56_pct']:>8.2f} %")
    print(f"  {'CV Tier Accuracy':<35} {cv_metrics['tier_classification_accuracy_pct']:>8.2f} %")
    print(f"  {'Overfitting Risk':<35} {overfit_analysis['overfitting_risk_level']:>10}")
    print(f"  {'Train-CV MAE Gap':<35} {overfit_analysis['train_vs_cv_mae_gap_cm']:>8.4f} cm")
    print(f"  {'Train-CV Tier Gap':<35} {overfit_analysis['train_vs_cv_tier_gap_pct']:>8.2f} %")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
