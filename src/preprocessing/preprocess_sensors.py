"""
Sensor Data Preprocessing Module
Extracts dynamic accelerations, applies Butterworth filtering,
extracts ground-truth pothole shock profiles, and computes windowed ML features.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew

# Target paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
SENSOR_INPUT_DIR = os.path.join(DATA_DIR, "sensor_data", "Pothole, Road Quality Detection Sensor data", "Pothole")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed", "sensors")

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a digital Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=1.0, highcut=20.0, fs=50.0, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    if len(data) <= 3 * order:
        return data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def estimate_gravity_and_dynamic(acc_x, acc_y, acc_z, fs=50.0):
    """
    Separate dynamic acceleration from static gravity vector.
    Calculates total dynamic magnitude and gravity-compensated vertical component.
    """
    total_acc = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Low-pass filter to estimate gravity magnitude (approx 1.0g)
    nyq = 0.5 * fs
    cutoff = min(0.5 / nyq, 0.99)
    b, a = butter(2, cutoff, btype='low')
    if len(total_acc) > 15:
        gravity_est = filtfilt(b, a, total_acc)
    else:
        gravity_est = np.ones_like(total_acc)
        
    dynamic_mag = np.abs(total_acc - gravity_est)
    return total_acc, gravity_est, dynamic_mag

def compute_fft_energy(signal, fs=50.0, low_freq=3.0, high_freq=12.0):
    """Compute spectral energy in the primary road shock frequency band."""
    n = len(signal)
    if n < 4:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    if len(idx) == 0:
        return 0.0
    return float(np.sum(fft_vals[idx]**2) / n)

def extract_window_features(win_df, fs=50.0):
    """Extract descriptive time-series & spectral features from a sensor window."""
    dyn_mag = win_df['dynamic_mag'].values
    dyn_filt = win_df['filtered_mag'].values
    speed = win_df['speed'].values if 'speed' in win_df.columns else np.zeros(len(win_df))
    
    # Gyro magnitude
    gyro_x = win_df['gyroX'].values if 'gyroX' in win_df.columns else np.zeros(len(win_df))
    gyro_y = win_df['gyroY'].values if 'gyroY' in win_df.columns else np.zeros(len(win_df))
    gyro_z = win_df['gyroZ'].values if 'gyroZ' in win_df.columns else np.zeros(len(win_df))
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    rms = np.sqrt(np.mean(dyn_filt**2)) if len(dyn_filt) > 0 else 0.0
    peak = np.max(np.abs(dyn_filt)) if len(dyn_filt) > 0 else 0.0
    p2p = (np.max(dyn_filt) - np.min(dyn_filt)) if len(dyn_filt) > 0 else 0.0
    crest_factor = (peak / (rms + 1e-6))
    
    kurt = float(kurtosis(dyn_filt)) if len(dyn_filt) > 4 else 0.0
    skewness = float(skew(dyn_filt)) if len(dyn_filt) > 4 else 0.0
    var = float(np.var(dyn_filt)) if len(dyn_filt) > 0 else 0.0
    
    fft_energy = compute_fft_energy(dyn_filt, fs=fs, low_freq=3.0, high_freq=12.0)
    
    return {
        'mean_speed': float(np.mean(speed)),
        'max_speed': float(np.max(speed)),
        'accel_rms': float(rms),
        'accel_peak': float(peak),
        'accel_p2p': float(p2p),
        'crest_factor': float(crest_factor),
        'kurtosis': kurt,
        'skewness': skewness,
        'variance': var,
        'gyro_rms': float(np.sqrt(np.mean(gyro_mag**2))),
        'gyro_peak': float(np.max(gyro_mag)) if len(gyro_mag) > 0 else 0.0,
        'fft_energy_3_12hz': float(fft_energy)
    }

def process_all_sensor_trips(window_duration=2.0, overlap=0.5, target_fs=50.0):
    """
    Main preprocessing pipeline for Pothole and Road Quality Sensor data.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sensor_files = sorted(glob.glob(os.path.join(SENSOR_INPUT_DIR, "*_sensors.csv")))
    
    all_windows = []
    all_shock_signatures = []
    
    print(f"Found {len(sensor_files)} trip sensor files in: {SENSOR_INPUT_DIR}")
    
    for s_file in sensor_files:
        trip_name = os.path.basename(s_file).replace("_sensors.csv", "")
        p_file = os.path.join(SENSOR_INPUT_DIR, f"{trip_name}_potholes.csv")
        
        print(f"Processing {trip_name}...")
        df_sensors = pd.read_csv(s_file).sort_values("timestamp").reset_index(drop=True)
        
        pothole_times = []
        if os.path.exists(p_file):
            df_pot = pd.read_csv(p_file)
            if 'timestamp' in df_pot.columns:
                pothole_times = df_pot['timestamp'].dropna().values
        
        # 1. Resample to uniform grid
        t_start = df_sensors['timestamp'].min()
        t_end = df_sensors['timestamp'].max()
        if (t_end - t_start) <= window_duration:
            continue
            
        uniform_t = np.arange(t_start, t_end, 1.0 / target_fs)
        
        # Interpolate numeric columns onto uniform timeline
        resampled_dict = {'timestamp': uniform_t}
        for col in ['latitude', 'longitude', 'speed', 'accelerometerX', 'accelerometerY', 'accelerometerZ', 'gyroX', 'gyroY', 'gyroZ']:
            if col in df_sensors.columns:
                resampled_dict[col] = np.interp(uniform_t, df_sensors['timestamp'], df_sensors[col])
        
        res_df = pd.DataFrame(resampled_dict)
        
        # 2. Gravity estimation and dynamic acceleration
        total_acc, grav, dyn_mag = estimate_gravity_and_dynamic(
            res_df['accelerometerX'].values,
            res_df['accelerometerY'].values,
            res_df['accelerometerZ'].values,
            fs=target_fs
        )
        res_df['dynamic_mag'] = dyn_mag
        res_df['filtered_mag'] = apply_bandpass(dyn_mag, lowcut=1.0, highcut=20.0, fs=target_fs)
        
        # 3. Extract exact ground truth pothole shock waveforms (-1.0s to +1.0s)
        for idx, pt in enumerate(pothole_times):
            shock_win = res_df[(res_df['timestamp'] >= pt - 1.0) & (res_df['timestamp'] <= pt + 1.0)]
            if len(shock_win) >= 10:
                peak_g = float(np.max(shock_win['filtered_mag']))
                p2p_g = float(np.max(shock_win['filtered_mag']) - np.min(shock_win['filtered_mag']))
                all_shock_signatures.append({
                    'trip_id': trip_name,
                    'pothole_idx': idx,
                    'pothole_timestamp': pt,
                    'peak_g': peak_g,
                    'p2p_g': p2p_g,
                    'duration_sec': 2.0,
                    'samples_count': len(shock_win),
                    'waveform_csv': ";".join([f"{v:.4f}" for v in shock_win['filtered_mag'].values])
                })
        
        # 4. Sliding window segmentation
        step = window_duration * (1.0 - overlap)
        cur_t = t_start
        win_idx = 0
        
        while cur_t + window_duration <= t_end:
            win_df = res_df[(res_df['timestamp'] >= cur_t) & (res_df['timestamp'] < cur_t + window_duration)]
            
            if len(win_df) >= int(target_fs * window_duration * 0.75):
                feats = extract_window_features(win_df, fs=target_fs)
                
                # Check if this window contains a labeled pothole
                has_pothole = 0
                for pt in pothole_times:
                    if cur_t <= pt < (cur_t + window_duration):
                        has_pothole = 1
                        break
                        
                feats.update({
                    'trip_id': trip_name,
                    'window_idx': win_idx,
                    'start_time': cur_t,
                    'end_time': cur_t + window_duration,
                    'label': has_pothole
                })
                all_windows.append(feats)
                win_idx += 1
                
            cur_t += step
            
    # Save outputs
    features_df = pd.DataFrame(all_windows)
    features_path = os.path.join(OUTPUT_DIR, "sensor_features_pothole.csv")
    features_df.to_csv(features_path, index=False)
    print(f"Saved {len(features_df)} window features to: {features_path}")
    if 'label' in features_df.columns:
        print(f"Class distribution: {features_df['label'].value_counts().to_dict()}")
        
    shocks_df = pd.DataFrame(all_shock_signatures)
    shocks_path = os.path.join(OUTPUT_DIR, "pothole_shock_signatures.csv")
    shocks_df.to_csv(shocks_path, index=False)
    print(f"Saved {len(shocks_df)} ground-truth pothole shock waveforms to: {shocks_path}")

if __name__ == "__main__":
    process_all_sensor_trips()

