"""
Two Wheeler Road Surface Classifier Preprocessing Module
Extracts 50 Hz filtered dynamics and road surface vibration features across:
Bitumin (Asphalt), Block (Pavers), Concrete, and Kanker (Gravel/Unpaved).
Output saved to: data/processed/sensors/two_wheeler/
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "sensor_data", "Two Wheeler Road Surface Classifier", "Raw Data")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "sensors", "two_wheeler")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=1.0, highcut=20.0, fs=50.0):
    if len(data) <= 12:
        return data
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def compute_fft_energy(signal, fs=50.0, low_freq=3.0, high_freq=12.0):
    n = len(signal)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    return float(np.sum(fft_vals[idx]**2) / n) if len(idx) > 0 else 0.0

def process_two_wheeler_dataset(window_sec=2.0, overlap=0.5, target_fs=50.0):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    classes = ['Bitumin', 'Block', 'Concrete', 'Kanker']
    
    all_features = []
    
    for cls_name in classes:
        cls_dir = os.path.join(INPUT_BASE, cls_name)
        csv_files = sorted(glob.glob(os.path.join(cls_dir, "*.csv")))
        print(f"Processing class '{cls_name}': {len(csv_files)} files...")
        
        for fpath in csv_files:
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath)
            except Exception:
                continue
                
            if len(df) < 50 or 'Timestamp' not in df.columns:
                continue
                
            # Convert timestamp ms to seconds
            t_ms = df['Timestamp'].values
            t_sec = (t_ms - t_ms[0]) / 1000.0
            
            t_total = t_sec[-1]
            if t_total <= window_sec:
                continue
                
            # Resample to uniform 50 Hz
            uniform_t = np.arange(0, t_total, 1.0 / target_fs)
            acc_x = np.interp(uniform_t, t_sec, df['X_Acc'].values)
            acc_y = np.interp(uniform_t, t_sec, df['Y_Acc'].values)
            acc_z = np.interp(uniform_t, t_sec, df['Z_Acc'].values)
            
            gyro_x = np.interp(uniform_t, t_sec, df['X_Gyro'].values)
            gyro_y = np.interp(uniform_t, t_sec, df['Y_Gyro'].values)
            gyro_z = np.interp(uniform_t, t_sec, df['Z_Gyro'].values)
            
            # Dynamic magnitude
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            # Remove approximate 9.81 m/s^2
            dyn_mag = np.abs(acc_mag - 9.81)
            filt_mag = apply_bandpass(dyn_mag, fs=target_fs)
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            
            # Slice into windows
            step = int(window_sec * (1.0 - overlap) * target_fs)
            win_size = int(window_sec * target_fs)
            
            for start_idx in range(0, len(uniform_t) - win_size, step):
                w_filt = filt_mag[start_idx:start_idx + win_size]
                w_gyro = gyro_mag[start_idx:start_idx + win_size]
                
                rms = np.sqrt(np.mean(w_filt**2))
                peak = np.max(np.abs(w_filt))
                p2p = np.max(w_filt) - np.min(w_filt)
                crest = peak / (rms + 1e-6)
                
                kurt = float(kurtosis(w_filt)) if len(w_filt) > 4 else 0.0
                sk = float(skew(w_filt)) if len(w_filt) > 4 else 0.0
                var = float(np.var(w_filt))
                fft_e = compute_fft_energy(w_filt, fs=target_fs)
                
                all_features.append({
                    'surface_class': cls_name,
                    'file_name': fname,
                    'window_start_sec': round(start_idx / target_fs, 2),
                    'accel_rms': float(rms),
                    'accel_peak': float(peak),
                    'accel_p2p': float(p2p),
                    'crest_factor': float(crest),
                    'kurtosis': kurt,
                    'skewness': sk,
                    'variance': var,
                    'gyro_rms': float(np.sqrt(np.mean(w_gyro**2))),
                    'gyro_peak': float(np.max(w_gyro)),
                    'fft_energy_3_12hz': float(fft_e)
                })
                
    out_df = pd.DataFrame(all_features)
    out_path = os.path.join(OUTPUT_BASE, "two_wheeler_surface_features.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Two-Wheeler Preprocessing Complete! Saved {len(out_df)} windows to: {out_path}")
    print(out_df['surface_class'].value_counts().to_dict())

if __name__ == "__main__":
    process_two_wheeler_dataset()

