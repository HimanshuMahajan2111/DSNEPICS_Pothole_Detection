"""
=============================================================================
SINGLE-WINDOW SENSOR DEPTH ESTIMATION INFERENCE DEMO
=============================================================================
Demonstrates how to pass an arbitrary IMU window to the engine to receive:
  - Metric Pothole Depth (cm)
  - Physical Width (cm)
  - Impact Energy (Joules)
  - Maintenance Severity Tier
=============================================================================
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from ml_sensor_depth import SensorMLDepthEngine

def demo_single_shock():
    engine = SensorMLDepthEngine()
    
    # Example simulated 2.0s pothole shock reading (sampling rate = 20 Hz, 40 points)
    # Peak-to-peak shock of ~0.65g with vehicle speed of 12 m/s (43 km/h)
    fs = 20.0
    t = np.linspace(-1.0, 1.0, int(2.0 * fs))
    # Characteristic pothole impact wave: negative dip (wheel falling) followed by sharp rebound spike
    acc_z_synthetic = -0.45 * np.exp(-((t + 0.1)**2) / 0.02) + 0.65 * np.exp(-((t - 0.1)**2) / 0.015) + np.random.normal(0, 0.03, len(t))
    
    speed_mps = 12.0  # 43.2 km/h
    
    print("\n" + "=" * 65)
    print("DEMO: INFERRING POTHOLE METRICS FROM A SENSOR ACCELERATION WINDOW")
    print("=" * 65)
    print(f"Sampling Rate : {fs} Hz")
    print(f"Vehicle Speed : {speed_mps * 3.6:.1f} km/h ({speed_mps:.1f} m/s)")
    print(f"Max Acceleration Shock PtP: {np.ptp(acc_z_synthetic):.2f} g")
    
    result = engine.predict(acc_z_synthetic, fs=fs, speed_mps=speed_mps, acc_in_g=True)
    
    print("\n--- ESTIMATED POTHOLE DAMAGE METRICS ---")
    print(f"  * Estimated Metric Depth: {result['predicted_depth_cm']:.2f} cm")
    print(f"  * Estimated Cavity Width: {result['estimated_width_cm']:.1f} cm")
    print(f"  * Dynamic Impact Energy : {result['impact_energy_j']:.2f} Joules")
    print(f"  * Severity Rating       : Tier {result['predicted_severity_tier']} ({result['tier_label']})")
    print("=" * 65 + "\n")
    return result

if __name__ == "__main__":
    demo_single_shock()
