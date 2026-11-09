"""
Intelligent Vehicle Perception (IVP) Preprocessing Module
Extracts 100 Hz suspension transfer dynamics and road condition features across:
Dashboard, Above Suspension, and Below Suspension sensor mounts.
Output saved to: data/processed/sensors/ivp_suspension/
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "sensor_data", "Intelligent Vehicle Perception")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "sensors", "ivp_suspension")

def butter_bandpass(lowcut=1.0, highcut=20.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, fs=100.0):
    if len(data) <= 15:
        return data
    b, a = butter_bandpass(1.0, 20.0, fs=fs)
    return filtfilt(b, a, data)

def process_ivp_trips(pvs_limit=5, window_sec=2.0, target_fs=100.0):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    pvs_dirs = sorted(glob.glob(os.path.join(INPUT_BASE, "PVS *")))
    if pvs_limit:
        pvs_dirs = pvs_dirs[:pvs_limit]
        
    print(f"Processing {len(pvs_dirs)} IVP trip directories...")
    all_features = []
    
    for p_dir in pvs_dirs:
        trip_name = os.path.basename(p_dir)
        imu_file = os.path.join(p_dir, "dataset_gps_mpu_left.csv")
        lbl_file = os.path.join(p_dir, "dataset_labels.csv")
        
        if not os.path.exists(imu_file) or not os.path.exists(lbl_file):
            continue
            
        print(f"Processing {trip_name}...")
        df_imu = pd.read_csv(imu_file)
        df_lbl = pd.read_csv(lbl_file)
        
        n_samples = min(len(df_imu), len(df_lbl))
        if n_samples < int(window_sec * target_fs):
            continue
            
        # Extract 3-point vertical accelerations (Z-axis is index of vertical shocks)
        z_dash = df_imu['acc_z_dashboard'].values[:n_samples]
        z_above = df_imu['acc_z_above_suspension'].values[:n_samples]
        z_below = df_imu['acc_z_below_suspension'].values[:n_samples]
        
        # Remove gravity offset (around 9.8) and filter
        filt_dash = apply_bandpass(np.abs(z_dash - np.mean(z_dash)), fs=target_fs)
        filt_above = apply_bandpass(np.abs(z_above - np.mean(z_above)), fs=target_fs)
        filt_below = apply_bandpass(np.abs(z_below - np.mean(z_below)), fs=target_fs)
        
        # Window slicing
        win_size = int(window_sec * target_fs)
        step = win_size // 2 # 50% overlap
        
        for idx in range(0, n_samples - win_size, step):
            w_dash = filt_dash[idx:idx + win_size]
            w_above = filt_above[idx:idx + win_size]
            w_below = filt_below[idx:idx + win_size]
            
            rms_dash = float(np.sqrt(np.mean(w_dash**2)))
            rms_below = float(np.sqrt(np.mean(w_below**2)))
            peak_below = float(np.max(w_below))
            
            # Suspension transmission ratio (how much wheel shock reaches car body)
            suspension_ratio = float(rms_dash / (rms_below + 1e-5))
            
            # Majority vote on labels in window
            w_lbl = df_lbl.iloc[idx:idx + win_size]
            paved = int(w_lbl['paved_road'].mean() > 0.5) if 'paved_road' in w_lbl else 1
            asphalt = int(w_lbl['asphalt_road'].mean() > 0.5) if 'asphalt_road' in w_lbl else 1
            speed_bump = int(w_lbl['speed_bump_asphalt'].mean() > 0.1 or w_lbl['speed_bump_cobblestone'].mean() > 0.1) if 'speed_bump_asphalt' in w_lbl else 0
            bad_road = int(w_lbl['bad_road_left'].mean() > 0.3) if 'bad_road_left' in w_lbl else 0
            
            all_features.append({
                'trip': trip_name,
                'window_sec': round(idx / target_fs, 2),
                'rms_dashboard': rms_dash,
                'rms_above_suspension': float(np.sqrt(np.mean(w_above**2))),
                'rms_below_suspension': rms_below,
                'peak_wheel_shock': peak_below,
                'suspension_isolation_ratio': suspension_ratio,
                'kurtosis_wheel': float(kurtosis(w_below)),
                'is_paved': paved,
                'is_asphalt': asphalt,
                'is_speed_bump': speed_bump,
                'is_bad_road': bad_road
            })
            
    df_out = pd.DataFrame(all_features)
    out_path = os.path.join(OUTPUT_BASE, "ivp_suspension_features.csv")
    df_out.to_csv(out_path, index=False)
    print(f"IVP Preprocessing Complete! Saved {len(df_out)} window features to: {out_path}")
    print("Road condition summary:", df_out[['is_paved', 'is_bad_road', 'is_speed_bump']].sum().to_dict())

if __name__ == "__main__":
    process_ivp_trips()

