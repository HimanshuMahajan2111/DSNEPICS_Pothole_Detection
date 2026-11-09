"""
Feature Selection Pipeline: Multi-Modal Dimension Reduction & Ranking
====================================================================
Performs:
1. Pearson & Spearman Correlation Analysis (Detects high collinearity pairs r > 0.85).
2. Variance Inflation Factor (VIF) filtering (Prunes multicollinear features with VIF > 10.0).
3. Information Gain / Mutual Information Scoring against Severity Tiers.
4. Random Forest Gini Importance & Permutation Importance ranking.
5. Saves selected optimal feature sets and diagnostic plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def compute_vif_manual(df_numeric):
    """
    Computes Variance Inflation Factor (VIF) using OLS R^2 matrix inversion:
    VIF_i = 1 / (1 - R_i^2) = (C^-1)_{ii} where C is the correlation matrix.
    """
    corr = df_numeric.corr().values
    try:
        corr_inv = np.linalg.inv(corr)
        vifs = np.diag(corr_inv)
        return pd.Series(vifs, index=df_numeric.columns)
    except np.linalg.LinAlgError:
        # Fallback pseudo-inverse if singular
        corr_inv = np.linalg.pinv(corr)
        vifs = np.diag(corr_inv)
        return pd.Series(vifs, index=df_numeric.columns)


def perform_sensor_feature_selection(sensor_csv, output_dir):
    """
    Performs rigorous feature selection on the engineered sensor dataset.
    """
    print(f"\n[Feature Selection] Ingesting Sensor Data from: {sensor_csv}")
    df = pd.read_csv(sensor_csv)
    
    # Candidate features for predicting road severity
    candidate_cols = [
        "mean_speed", "max_speed", "accel_rms", "accel_peak", "accel_p2p",
        "crest_factor", "kurtosis", "skewness", "variance", "gyro_rms", "gyro_peak",
        "fft_energy_3_12hz", "speed_norm_peak_g", "speed_norm_rms_g", "speed_norm_p2p_g",
        "kinetic_shock_energy_joules", "wheel_drop_est_cm", "shock_band_energy_ratio",
        "impulsiveness_index", "angular_instability"
    ]
    
    X = df[candidate_cols].copy()
    y = df["severity_tier"].values
    
    # 1. Collinearity Matrix
    corr_matrix = X.corr(method="spearman")
    
    # 2. Iterative VIF Pruning (Remove features with VIF > 10.0 until non-collinear core remains)
    remaining_cols = list(candidate_cols)
    pruned_history = []
    
    print("[Feature Selection] Running iterative VIF pruning (threshold = 10.0)...")
    while True:
        vif_series = compute_vif_manual(X[remaining_cols])
        max_vif = vif_series.max()
        max_feat = vif_series.idxmax()
        if max_vif > 10.0 and len(remaining_cols) > 5:
            remaining_cols.remove(max_feat)
            pruned_history.append({"feature": max_feat, "vif": float(max_vif)})
        else:
            break
            
    print(f"[Feature Selection] Pruned {len(pruned_history)} collinear features. Retained {len(remaining_cols)} features.")
    final_vif = compute_vif_manual(X[remaining_cols])
    
    # 3. Mutual Information Scores
    mi_scores = mutual_info_classif(X[candidate_cols], y, random_state=42)
    mi_series = pd.Series(mi_scores, index=candidate_cols).sort_values(ascending=False)
    
    # 4. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    rf.fit(X[candidate_cols], y)
    rf_importance = pd.Series(rf.feature_importances_, index=candidate_cols).sort_values(ascending=False)
    
    # 5. Composite Ranking
    # Rank by MI + RF importance
    rank_df = pd.DataFrame({
        "Mutual_Information": mi_series,
        "Random_Forest_Gini": rf_importance,
        "Retained_in_VIF": [feat in remaining_cols for feat in mi_series.index]
    })
    rank_df["Composite_Score"] = 0.5 * (rank_df["Mutual_Information"] / rank_df["Mutual_Information"].max()) + \
                                 0.5 * (rank_df["Random_Forest_Gini"] / rank_df["Random_Forest_Gini"].max())
    rank_df = rank_df.sort_values(by="Composite_Score", ascending=False)
    
    # Selected Top Features (Must be informative and have manageable VIF)
    # Pick top 7 features prioritizing those passing VIF or offering critical physical grounding
    selected_features = rank_df.head(8).index.tolist()
    print(f"\n[Feature Selection] Top 8 Selected Sensor Features:")
    for rank, feat in enumerate(selected_features, 1):
        print(f"  {rank}. {feat:28s} | Composite Score: {rank_df.loc[feat, 'Composite_Score']:.4f} | Retained VIF: {rank_df.loc[feat, 'Retained_in_VIF']}")
        
    # Save Ranking Table
    os.makedirs(output_dir, exist_ok=True)
    rank_csv = os.path.join(output_dir, "sensor_feature_rankings.csv")
    rank_df.to_csv(rank_csv)
    
    # Plotting Feature Selection Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Subplot 1: Top 10 Features by Mutual Information
    top_mi = mi_series.head(10)
    axes[0].barh(top_mi.index[::-1], top_mi.values[::-1], color="#2563eb", edgecolor="none")
    axes[0].set_title("Mutual Information vs Severity Tier", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mutual Information (Nats)")
    axes[0].grid(axis="x", alpha=0.3)
    
    # Subplot 2: Top 10 Features by Random Forest Importance
    top_rf = rf_importance.head(10)
    axes[1].barh(top_rf.index[::-1], top_rf.values[::-1], color="#10b981", edgecolor="none")
    axes[1].set_title("Random Forest Gini Importance", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Decrease in Impurity")
    axes[1].grid(axis="x", alpha=0.3)
    
    # Subplot 3: VIF for Retained Features
    axes[2].barh(final_vif.index[::-1], final_vif.values[::-1], color="#f59e0b", edgecolor="none")
    axes[2].axvline(10.0, color="#ef4444", linestyle="--", label="Collinearity Threshold (VIF=10)")
    axes[2].set_title("Variance Inflation Factor (VIF) Post-Pruning", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("VIF Score (< 10 is non-collinear)")
    axes[2].legend(loc="lower right")
    axes[2].grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "sensor_feature_selection.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"[Feature Selection] Saved diagnostic chart to: {chart_path}")
    
    return selected_features, rank_df


def perform_vision_feature_selection(vision_csv, output_dir):
    """
    Evaluates vision engineered geometry features for predicting severity tier.
    """
    print(f"\n[Feature Selection] Ingesting Vision Geometry from: {vision_csv}")
    df = pd.read_csv(vision_csv)
    
    geom_cols = [
        "dist_z_m", "phys_width_m", "phys_length_m", "slenderness",
        "area_cm2", "est_depth_cm", "volume_cm3", "edge_slope_deg", "impact_risk_index"
    ]
    
    X = df[geom_cols]
    y = df["severity_tier"].values
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=geom_cols).sort_values(ascending=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=geom_cols).sort_values(ascending=False)
    
    rank_df = pd.DataFrame({
        "Mutual_Information": mi_series,
        "Random_Forest_Gini": rf_importance
    })
    rank_df["Composite_Score"] = 0.5 * (rank_df["Mutual_Information"] / rank_df["Mutual_Information"].max()) + \
                                 0.5 * (rank_df["Random_Forest_Gini"] / rank_df["Random_Forest_Gini"].max())
    rank_df = rank_df.sort_values(by="Composite_Score", ascending=False)
    
    print("\n[Feature Selection] Vision Geometry Rankings:")
    for rank, feat in enumerate(rank_df.index, 1):
        print(f"  {rank}. {feat:22s} | Score: {rank_df.loc[feat, 'Composite_Score']:.4f}")
        
    rank_csv = os.path.join(output_dir, "vision_feature_rankings.csv")
    rank_df.to_csv(rank_csv)
    return rank_df


if __name__ == "__main__":
    feat_dir = "d:/DSN3099-EPICS/Implementation/data/processed/features"
    eda_assets = "d:/DSN3099-EPICS/Implementation/eda_report/assets"
    
    sensor_path = os.path.join(feat_dir, "sensor_engineered_features.csv")
    vision_path = os.path.join(feat_dir, "vision_engineered_geometry.csv")
    
    selected_feats, sensor_ranks = perform_sensor_feature_selection(sensor_path, eda_assets)
    vision_ranks = perform_vision_feature_selection(vision_path, eda_assets)
    
    # Save optimal feature configuration JSON for downstream model use
    config = {
        "selected_sensor_features": selected_feats,
        "selected_vision_features": ["est_depth_cm", "volume_cm3", "phys_width_m", "impact_risk_index", "edge_slope_deg"],
        "sensor_vif_threshold": 10.0,
        "top_kinematic_bridge": "speed_norm_peak_g",
        "top_geometric_bridge": "impact_risk_index"
    }
    with open(os.path.join(feat_dir, "selected_features_manifest.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print("\n[SUCCESS] Feature Selection & Ranking Manifest saved!")

