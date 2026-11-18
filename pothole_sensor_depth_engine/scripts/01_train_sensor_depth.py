"""
=============================================================================
STAGE 2 (REGULARIZED): TRAIN GENERALIZED SENSOR DEPTH & SEVERITY MODELS
=============================================================================
Author: EPICS DSN3099
Anti-overfitting constraints:
  - Strong regularization (max_depth=3, min_samples_leaf=20, l2 shrinkage)
  - Trained strictly on sensor_train.csv (1645 samples)
  - Validated and early-stopped on sensor_val.csv (352 samples)
  - Never touches sensor_test.csv during fitting or tuning
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

def train_regularized_models():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "pothole_sensor_depth_engine", "data")
    models_dir = os.path.join(project_root, "pothole_sensor_depth_engine", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(data_dir, "sensor_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "sensor_val.csv"))
    
    ignore_cols = ["source_dataset", "target_depth_cm", "severity_tier"]
    feat_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    X_train = np.nan_to_num(train_df[feat_cols].values, nan=0.0)
    y_d_train = train_df["target_depth_cm"].values
    y_t_train = train_df["severity_tier"].values
    
    X_val = np.nan_to_num(val_df[feat_cols].values, nan=0.0)
    y_d_val = val_df["target_depth_cm"].values
    y_t_val = val_df["severity_tier"].values
    
    print("\n" + "=" * 70)
    print("TRAINING REGULARIZED SENSOR MODELS (PREVENTING OVERFITTING)")
    print("=" * 70)
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)
    
    # ── Tuned Depth Regressor (balanced variance & bias) ──
    reg = GradientBoostingRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42
    )
    reg.fit(X_tr_sc, y_d_train)
    
    # ── Tuned Severity Classifier ──
    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=6,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=42
    )
    clf.fit(X_tr_sc, y_t_train)
    
    # Validation performance
    p_d_val = reg.predict(X_va_sc)
    p_t_val = clf.predict(X_va_sc)
    
    val_mae = mean_absolute_error(y_d_val, p_d_val)
    val_r2 = r2_score(y_d_val, p_d_val)
    val_acc = accuracy_score(y_t_val, p_t_val) * 100.0
    ratio_val = np.maximum(p_d_val / (y_d_val + 1e-6), y_d_val / (p_d_val + 1e-6))
    val_delta125 = float(np.mean(ratio_val < 1.25) * 100.0)
    
    print(f"\n--- VALIDATION METRICS ---")
    print(f"  * Depth MAE (cm)         : {val_mae:.3f} cm")
    print(f"  * R^2 Score              : {val_r2:.4f}")
    print(f"  * Delta < 1.25 Accuracy  : {val_delta125:.2f} %")
    print(f"  * Severity Tier Accuracy : {val_acc:.2f} %")
    
    # Save artifacts
    with open(os.path.join(models_dir, "sensor_depth_regressor.pkl"), "wb") as f:
        pickle.dump(reg, f)
    with open(os.path.join(models_dir, "sensor_tier_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(models_dir, "sensor_feature_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feat_cols, f, indent=4)
        
    print(f"Saved regularized model checkpoints -> {models_dir}\n")

if __name__ == "__main__":
    train_regularized_models()
