"""
Sensor Data Cleaning Module
Purges NaNs, filters stationary vehicle idling windows, and imputes missing metadata.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
SENSORS_BASE = os.path.join(DATA_DIR, "processed", "sensors")

def clean_roadsens_4m():
    """Purge 66 NaN rows from roadsens_anomaly_features.csv"""
    fpath = os.path.join(SENSORS_BASE, "roadsens_4m", "roadsens_anomaly_features.csv")
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return
        
    df = pd.read_csv(fpath)
    orig_len = len(df)
    
    # Drop rows where any numerical acceleration feature is NaN
    clean_df = df.dropna(subset=['accel_rms', 'accel_peak', 'accel_p2p']).copy()
    purged_cnt = orig_len - len(clean_df)
    
    # Overwrite in place and save clean copy
    clean_df.to_csv(fpath, index=False)
    clean_df.to_csv(fpath.replace(".csv", "_clean.csv"), index=False)
    print(f"[roadsens_4m] Cleaned NaNs: Purged {purged_cnt} rows. Clean rows: {len(clean_df):,} (from {orig_len:,})")

def clean_pothole_trips(min_speed_mps=1.0):
    """
    Filter out 208 stationary vehicle idling windows (speed < 1.0 m/s)
    to eliminate engine idle vibration from road roughness baselines.
    """
    fpath = os.path.join(SENSORS_BASE, "pothole_trips", "sensor_features_pothole.csv")
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return
        
    df = pd.read_csv(fpath)
    orig_len = len(df)
    
    # Identify idling windows: speed < min_speed_mps and no pothole impact
    idling_mask = (df['mean_speed'] < min_speed_mps) & (df['label'] == 0)
    idling_cnt = int(idling_mask.sum())
    
    clean_df = df[~idling_mask].copy().reset_index(drop=True)
    
    clean_path = os.path.join(SENSORS_BASE, "pothole_trips", "sensor_features_pothole_clean.csv")
    clean_df.to_csv(clean_path, index=False)
    # Also update main file
    clean_df.to_csv(fpath, index=False)
    
    print(f"[pothole_trips] Filtered Idling Noise: Removed {idling_cnt} stationary windows (speed < {min_speed_mps} m/s).")
    print(f"                Clean rows: {len(clean_df):,} (from {orig_len:,}). Labels: {clean_df['label'].value_counts().to_dict()}")

def clean_smart_road_sense():
    """Impute missing quality metadata with 'Unrated' in smart_road_sense_sample.csv"""
    fpath = os.path.join(SENSORS_BASE, "smart_road_sense", "smart_road_sense_sample.csv")
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        return
        
    df = pd.read_csv(fpath)
    missing_cnt = df['quality'].isna().sum() if 'quality' in df.columns else 0
    
    if 'quality' in df.columns:
        df['quality'] = df['quality'].fillna("Unrated")
        
    df.to_csv(fpath, index=False)
    print(f"[smart_road_sense] Imputed {missing_cnt:,} missing 'quality' records with 'Unrated'. Clean rows: {len(df):,}")

def run_sensor_cleaning():
    print("=" * 60)
    print("EXECUTING SENSOR DATA CLEANING PIPELINE")
    print("=" * 60)
    clean_roadsens_4m()
    clean_pothole_trips()
    clean_smart_road_sense()
    print("=" * 60)
    print("SENSOR DATA CLEANING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_sensor_cleaning()

