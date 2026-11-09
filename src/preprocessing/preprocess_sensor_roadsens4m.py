"""
RoadSens-4M Preprocessing Module
Extracts IMU vibrations, GIS elevation dynamics, and anomaly labels.
Output saved to: data/processed/sensors/roadsens_4m/
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "combined", "RoadSens-4M", "Combined CSV with GIS and Weather Data")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "sensors", "roadsens_4m")

def apply_bandpass(data, fs=50.0):
    if len(data) <= 12:
        return data
    b, a = butter(4, [0.04, 0.8], btype='band')
    return filtfilt(b, a, data)

def process_roadsens4m(max_files=40):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    anomaly_files = glob.glob(os.path.join(INPUT_BASE, "Road Anomalies", "**", "*.csv"), recursive=True)
    normal_files = glob.glob(os.path.join(INPUT_BASE, "Normal Road (No Annotation)", "**", "*.csv"), recursive=True)
    
    print(f"Found {len(anomaly_files)} anomaly files, {len(normal_files)} normal files in RoadSens-4M...")
    
    selected_files = [(f, 1) for f in anomaly_files[:max_files]] + [(f, 0) for f in normal_files[:max_files]]
    
    records = []
    for fpath, is_anomaly in selected_files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
            
        if len(df) < 50 or 'accelerometer_z' not in df.columns:
            continue
            
        acc_z = df['accelerometer_z'].values
        acc_y = df['accelerometer_y'].values if 'accelerometer_y' in df.columns else np.zeros_like(acc_z)
        acc_x = df['accelerometer_x'].values if 'accelerometer_x' in df.columns else np.zeros_like(acc_z)
        
        dyn_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        filt_mag = apply_bandpass(dyn_mag)
        
        elevation = df['GIS_Elevation'].values if 'GIS_Elevation' in df.columns else np.zeros_like(acc_z)
        
        # 50 samples (~1s at 50Hz)
        win_size = 50
        for i in range(0, len(filt_mag) - win_size, win_size):
            w = filt_mag[i:i + win_size]
            w_elev = elevation[i:i + win_size]
            
            records.append({
                'file': os.path.basename(fpath),
                'is_anomaly': is_anomaly,
                'accel_rms': float(np.sqrt(np.mean(w**2))),
                'accel_peak': float(np.max(np.abs(w))),
                'accel_p2p': float(np.max(w) - np.min(w)),
                'elevation_delta': float(np.max(w_elev) - np.min(w_elev)) if len(w_elev) > 0 else 0.0
            })
            
    df_out = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_BASE, "roadsens_anomaly_features.csv")
    df_out.to_csv(out_path, index=False)
    print(f"RoadSens-4M Preprocessing Complete! Saved {len(df_out)} rows to: {out_path}")
    print(df_out['is_anomaly'].value_counts().to_dict())

if __name__ == "__main__":
    process_roadsens4m()

