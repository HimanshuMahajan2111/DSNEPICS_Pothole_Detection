"""
=============================================================================
STEP 2: REAL-TIME CONTINUOUS SENSOR STREAM POTHOLE DEPTH DETECTOR
=============================================================================
Author: EPICS DSN3099
Engine Capabilities:
  - Consumes raw IMU streams (accelerometer, gyroscope, vehicle speed)
  - Continuously monitors vertical dynamic shock & energy thresholds
  - Automatically isolates pothole events with Quarter-Car kinematics
  - Computes calibrated metric depth, width, impact energy, and severity
  - Visualizes real-time sensor waveforms and diagnostic telemetry
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_sensor_depth import SensorMLDepthEngine

class RealtimeSensorPotholeDetector:
    def __init__(self, models_dir=None):
        self.ml_engine = SensorMLDepthEngine(models_dir=models_dir)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.output_dir = os.path.join(self.project_root, "pothole_sensor_depth_engine", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def scan_trip_stream(self, sensor_csv, window_sec=2.0, step_sec=0.5, shock_thresh_g=0.25):
        """
        Sliding-window stream processor across continuous accelerometer trip.
        Detects road cavities and outputs calibrated metric depths.
        """
        print(f"\nScanning trip sensor stream: {sensor_csv}")
        df = pd.read_csv(sensor_csv).sort_values("timestamp").reset_index(drop=True)
        
        # Sampling frequency
        dt = df["timestamp"].diff().median()
        fs = 1.0 / dt if dt > 0 else 5.0
        win_size = int(window_sec * fs)
        step_size = max(1, int(step_sec * fs))
        
        detected_events = []
        n_samples = len(df)
        
        t_arr = df["timestamp"].values
        az_arr = df["accelerometerZ"].values
        sp_arr = df["speed"].values if "speed" in df.columns else np.full(n_samples, 10.0)
        lat_arr = df["latitude"].values if "latitude" in df.columns else np.zeros(n_samples)
        lon_arr = df["longitude"].values if "longitude" in df.columns else np.zeros(n_samples)
        
        for start_idx in range(0, n_samples - win_size, step_size):
            end_idx = start_idx + win_size
            sub_az = az_arr[start_idx:end_idx]
            
            # Fast shock pre-filter (ignore smooth cruising roads)
            ptp = np.ptp(sub_az)
            if ptp < shock_thresh_g:
                continue
                
            sub_t = t_arr[start_idx:end_idx]
            center_t = sub_t[len(sub_t) // 2]
            avg_speed = float(np.nanmean(sp_arr[start_idx:end_idx]))
            if np.isnan(avg_speed) or avg_speed < 1.0:
                avg_speed = 10.0
                
            pred = self.ml_engine.predict(sub_az, fs=fs, speed_mps=avg_speed, acc_in_g=True)
            
            # Non-maximum suppression: prevent duplicate hits for the same pothole
            if detected_events and abs(center_t - detected_events[-1]["timestamp"]) < 1.0:
                if pred["predicted_depth_cm"] > detected_events[-1]["depth_cm"]:
                    detected_events[-1] = {
                        "timestamp": center_t,
                        "latitude": float(lat_arr[start_idx + win_size // 2]),
                        "longitude": float(lon_arr[start_idx + win_size // 2]),
                        "depth_cm": pred["predicted_depth_cm"],
                        "width_cm": pred["estimated_width_cm"],
                        "energy_j": pred["impact_energy_j"],
                        "ptp_acc": pred["peak_to_peak_acc"],
                        "tier": pred["predicted_severity_tier"],
                        "tier_label": pred["tier_label"],
                        "window_az": sub_az.tolist()
                    }
                continue
                
            event_dict = {
                "timestamp": center_t,
                "latitude": float(lat_arr[start_idx + win_size // 2]),
                "longitude": float(lon_arr[start_idx + win_size // 2]),
                "depth_cm": pred["predicted_depth_cm"],
                "width_cm": pred["estimated_width_cm"],
                "energy_j": pred["impact_energy_j"],
                "ptp_acc": pred["peak_to_peak_acc"],
                "tier": pred["predicted_severity_tier"],
                "tier_label": pred["tier_label"],
                "window_az": sub_az.tolist()
            }
            detected_events.append(event_dict)
            
        print(f"Detected {len(detected_events)} significant pothole shock events.")
        return detected_events, df

    def visualize_trip_telemetry(self, trip_name, events, df_stream):
        """
        Generates full-trip continuous acceleration telemetry with detected pothole depth markers.
        """
        fig = plt.figure(figsize=(16, 10), dpi=150)
        plt.subplots_adjust(hspace=0.35)
        
        t_rel = df_stream["timestamp"].values - df_stream["timestamp"].values[0]
        az = df_stream["accelerometerZ"].values
        speed = df_stream["speed"].values if "speed" in df_stream.columns else np.zeros(len(az))
        
        # Panel 1: Continuous Vertical Acceleration Waveform & Pothole Triggers
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(t_rel, az, color="#2b5c8f", lw=1.2, label="Vertical Acceleration Az (g)")
        ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.6)
        
        tier_colors = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
        
        for ev in events:
            ev_t = ev["timestamp"] - df_stream["timestamp"].values[0]
            c = tier_colors.get(ev["tier"], "red")
            ax1.axvline(ev_t, color=c, linestyle=":", alpha=0.85, lw=1.5)
            ax1.scatter([ev_t], [ev["ptp_acc"] * 0.5], color=c, s=55, zorder=5)
            ax1.text(ev_t, max(az) * 0.85, f"{ev['depth_cm']:.1f}cm", color=c, fontsize=8,
                     fontweight="bold", rotation=90, va="bottom", ha="center")
                     
        ax1.set_title(f"Trip Telemetry: Continuous Vertical IMU Stream & Detected Pothole Depths [{trip_name}]",
                      fontsize=12, fontweight="bold")
        ax1.set_ylabel("Acceleration Az (g)", fontsize=10, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")
        
        # Panel 2: Calibrated Road Cavity Depth & Width Scatter
        ax2 = fig.add_subplot(3, 1, 2)
        if events:
            ev_ts = [ev["timestamp"] - df_stream["timestamp"].values[0] for ev in events]
            ev_depths = [ev["depth_cm"] for ev in events]
            ev_widths = [ev["width_cm"] for ev in events]
            ev_colors = [tier_colors.get(ev["tier"], "blue") for ev in events]
            
            bars = ax2.bar(ev_ts, ev_depths, width=4.0, color=ev_colors, alpha=0.85, edgecolor="black")
            ax2.axhline(3.2, color="orange", linestyle="--", lw=1.5, label="Tier 2 Threshold (3.2 cm)")
            ax2.axhline(6.5, color="red", linestyle="--", lw=1.5, label="Tier 3 Critical (6.5 cm)")
            
            for b, d, w in zip(bars, ev_depths, ev_widths):
                ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.15,
                         f"d={d:.1f}\nw={w:.0f}cm", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
                         
        ax2.set_title("Identified Pothole Depth Profile & Maintenance Priority Classification", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Depression Depth (cm)", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, max([ev["depth_cm"] for ev in events] + [8.0]) + 1.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")
        
        # Panel 3: Vehicle Speed & Suspension Impact Energy Profile
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t_rel, speed * 3.6, color="#4daf4a", lw=1.5, label="Vehicle Speed (km/h)")
        ax3.set_ylabel("Speed (km/h)", color="#4daf4a", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Elapsed Trip Time (seconds)", fontsize=10, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        
        # Dual axis for suspension shock energy
        if events:
            ax3_r = ax3.twinx()
            ev_ts = [ev["timestamp"] - df_stream["timestamp"].values[0] for ev in events]
            ev_energies = [ev["energy_j"] for ev in events]
            ax3_r.stem(ev_ts, ev_energies, linefmt="m--", markerfmt="m^", basefmt="none", label="Shock Energy (Joules)")
            ax3_r.set_ylabel("Suspension Impact Energy (Joules)", color="m", fontsize=10, fontweight="bold")
            
        out_plot = os.path.join(self.output_dir, f"sensor_depth_telemetry_{trip_name}.png")
        plt.savefig(out_plot, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved telemetry diagnostic plot -> {out_plot}")
        return out_plot

def run_demo():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    detector = RealtimeSensorPotholeDetector()
    
    trip_path = os.path.join(project_root, "data", "sensor_data", "Pothole, Road Quality Detection Sensor data", "Pothole", "trip1_sensors.csv")
    events, df_stream = detector.scan_trip_stream(trip_path)
    
    # Generate visualization
    detector.visualize_trip_telemetry("trip1", events, df_stream)
    
    # Save JSON manifest of detected events
    manifest = os.path.join(detector.output_dir, "detected_trip1_potholes.json")
    with open(manifest, "w") as f:
        # Strip long arrays for clean json output
        clean_events = [{k: v for k, v in ev.items() if k != "window_az"} for ev in events]
        json.dump(clean_events, f, indent=4)
    print(f"Saved JSON audit manifest -> {manifest}")

if __name__ == "__main__":
    run_demo()
