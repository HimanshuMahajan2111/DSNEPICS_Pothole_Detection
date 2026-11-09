"""
Sensor Data v2 Preprocessing Module
Extracts mobile phone accelerometer features across 99 trips for:
Bumpy, Rough, and Smooth road conditions.
Output saved to: data/processed/sensors/sensor_data_v2/
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "sensor_data", "sensor_data_v2")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "sensors", "sensor_data_v2")

def apply_filter(data):
    if len(data) <= 12:
        return data
    b, a = butter(4, [0.05, 0.8], btype='band')
    return filtfilt(b, a, data)

def process_sensor_v2():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_BASE, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files in sensor_data_v2...")
    
    records = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
            
        time_col = [c for c in df.columns if 'time' in c.lower()]
        road_col = [c for c in df.columns if 'road' in c.lower() or 'type' in c.lower()]
        
        if not time_col or not road_col or len(df) < 30:
            continue
            
        tc = time_col[0]
        rc = road_col[0]
        
        # Pick filtered axis columns if present, else raw
        z_col = 'Z-axis' if 'Z-axis' in df.columns else ('Z-raw' if 'Z-raw' in df.columns else df.columns[3])
        y_col = 'Y-axis' if 'Y-axis' in df.columns else ('Y-raw' if 'Y-raw' in df.columns else df.columns[2])
        x_col = 'X-axis' if 'X-axis' in df.columns else ('X-raw' if 'X-raw' in df.columns else df.columns[1])
        
        z_vals = pd.to_numeric(df[z_col], errors='coerce').fillna(0).values
        y_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values
        x_vals = pd.to_numeric(df[x_col], errors='coerce').fillna(0).values
        
        mag = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
        filt = apply_filter(mag)
        
        road_type = str(df[rc].mode()[0]) if len(df[rc].dropna()) > 0 else "Unknown"
        
        # Window of 50 samples
        win_size = 50
        for i in range(0, len(filt) - win_size, win_size // 2):
            w = filt[i:i + win_size]
            records.append({
                'file': os.path.basename(f),
                'road_type': road_type,
                'accel_rms': float(np.sqrt(np.mean(w**2))),
                'accel_peak': float(np.max(np.abs(w))),
                'accel_p2p': float(np.max(w) - np.min(w)),
                'kurtosis': float(kurtosis(w)) if len(w) > 4 else 0.0,
                'variance': float(np.var(w))
            })
            
    df_out = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_BASE, "mobile_road_roughness.csv")
    df_out.to_csv(out_path, index=False)
    print(f"sensor_data_v2 Preprocessing Complete! Saved {len(df_out)} rows to: {out_path}")
    if 'road_type' in df_out:
        print(df_out['road_type'].value_counts().to_dict())

if __name__ == "__main__":
    process_sensor_v2()

