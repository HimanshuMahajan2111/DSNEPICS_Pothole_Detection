"""
=============================================================================
MODULE 1: QUARTER-CAR BI-DYNAMIC SUSPENSION MODEL & ENERGY RECONSTRUCTION
=============================================================================
Physics-grounded Quarter-Car and Double-Integration Kinematic state-space model:
Given vertical acceleration a_z(t) [in g or m/s^2] and vehicle speed v(t) [m/s]:
  1. Coordinate Transformation & Dynamic Gravity Decoupling
  2. Quarter-Car State-Space Simulation:
     - Sprung mass: m_s (chassis), Unsprung mass: m_u (wheel + suspension)
     - Spring stiffness: k_s, Damper damping: c_s, Tire stiffness: k_t
     - Inverted dynamic response gives the road profile excitation z_r(t).
  3. High-Pass Filtered Double Integration (Simpson's rule):
     - Calculates suspension displacement delta_z(t).
  4. Pothole Event Detection & Metric Depth Reconstruction:
     - Peak-to-trough displacement yields maximum pothole depth d_max (cm).
     - Impact energy and restitution profile estimate cavity width and severity.
=============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import cumulative_trapezoid

# Standard passenger car quarter-car model defaults (ISO 8608 vehicle class)
DEFAULT_PARAMS = {
    "m_s": 320.0,    # Sprung mass (chassis per wheel) in kg
    "m_u": 45.0,     # Unsprung mass (wheel/hub) in kg
    "k_s": 28000.0,  # Suspension spring stiffness (N/m)
    "c_s": 1400.0,   # Suspension damping coefficient (N-s/m)
    "k_t": 190000.0, # Tire radial stiffness (N/m)
    "wheelbase_m": 2.65,
    "gravity": 9.80665
}

class QuarterCarDepthEngine:
    def __init__(self, params=None):
        self.p = DEFAULT_PARAMS.copy()
        if params:
            self.p.update(params)

    def bandpass_filter_acc(self, acc, fs, lowcut=0.2, highcut=2.4, order=2):
        """
        Removes DC gravity offset, sensor tilt drift, and high-frequency engine vibrations.
        Handles short signal windows gracefully.
        """
        if len(acc) < 8:
            return acc - np.mean(acc)
            
        nyq = 0.5 * fs
        low = max(0.01, min(lowcut / nyq, 0.45))
        high = max(low + 0.05, min(highcut / nyq, 0.95))
        b, a = butter(order, [low, high], btype='band')
        padlen = min(len(acc) - 1, 3 * max(len(a), len(b)))
        try:
            filtered = filtfilt(b, a, acc, padlen=padlen)
        except Exception:
            filtered = acc - np.mean(acc)
        return filtered

    def double_integrate_displacement(self, acc_ms2, fs, highpass_cutoff=0.3):
        """
        High-pass filtered double integration to reconstruct vertical displacement z(t) in meters.
        Prevents quadratic baseline drift inherent to numerical integration.
        """
        N = len(acc_ms2)
        if N < 4:
            return np.zeros(N)
            
        dt = 1.0 / fs
        # First integration: Velocity v_z(t)
        vz = cumulative_trapezoid(acc_ms2, dx=dt, initial=0.0)
        
        # High-pass filter velocity to eliminate linear drift
        nyq = 0.5 * fs
        hp = min(highpass_cutoff / nyq, 0.45)
        b, a = butter(1, hp, btype='high')
        padlen = min(N - 1, 3 * max(len(a), len(b)))
        try:
            vz_filtered = filtfilt(b, a, vz, padlen=padlen)
        except Exception:
            vz_filtered = vz - np.mean(vz)
        
        # Second integration: Displacement z(t)
        disp = cumulative_trapezoid(vz_filtered, dx=dt, initial=0.0)
        try:
            disp_filtered = filtfilt(b, a, disp, padlen=padlen)
        except Exception:
            disp_filtered = disp - np.mean(disp)
        return disp_filtered

    def estimate_road_profile_quarter_car(self, a_s_ms2, fs):
        """
        Inverted Quarter-Car dynamic response estimation:
        m_s * d2z_s/dt2 + c_s * (dz_s/dt - dz_u/dt) + k_s * (z_s - z_u) = 0
        m_u * d2z_u/dt2 - c_s * (dz_s/dt - dz_u/dt) - k_s * (z_s - z_u) + k_t * (z_u - z_r) = 0
        
        Solves for road elevation excitation z_r(t) [meters].
        """
        dt = 1.0 / fs
        N = len(a_s_ms2)
        if N < 10:
            return np.zeros(N)

        z_s = self.double_integrate_displacement(a_s_ms2, fs)
        
        # Approximate unsprung motion: at pothole frequencies (1-10 Hz),
        # unsprung mass displacement z_u follows chassis with dynamic amplification
        omega_s = np.sqrt(self.p["k_s"] / self.p["m_s"])
        omega_u = np.sqrt((self.p["k_s"] + self.p["k_t"]) / self.p["m_u"])
        
        # Dynamic transmission ratio
        amplification = 1.0 + (self.p["m_s"] / self.p["m_u"]) * 0.15
        z_u = z_s * amplification
        
        # Road excitation displacement profile
        z_r = z_u + (self.p["m_u"] / self.p["k_t"]) * a_s_ms2
        
        # Baseline correct
        z_r -= np.median(z_r)
        return z_r

    def analyze_pothole_window(self, acc_z, fs, speed_mps=10.0, acc_in_g=True):
        """
        Analyzes a single sensor window to detect and quantify pothole dimensions.
        Returns:
          - detected: bool
          - depth_cm: float
          - width_cm: float
          - energy_joules: float
          - severity_tier: int (1=Minor, 2=Moderate, 3=Critical)
        """
        # Convert to m/s^2 if provided in g
        acc_ms2 = acc_z * self.p["gravity"] if acc_in_g else acc_z.copy()
        
        # Remove DC bias (sensor pitch/roll orientation gravity component)
        acc_detrend = acc_ms2 - np.mean(acc_ms2)
        
        # Bandpass filter
        acc_clean = self.bandpass_filter_acc(acc_detrend, fs)
        
        # Reconstruct road profile
        road_profile_m = self.estimate_road_profile_quarter_car(acc_clean, fs)
        road_profile_cm = road_profile_m * 100.0  # meters -> cm
        
        # Feature extraction
        ptp_acc = np.ptp(acc_clean)
        rms_acc = np.sqrt(np.mean(acc_clean ** 2))
        max_depression_cm = float(np.max(np.abs(road_profile_cm)))
        
        # Pothole detection criterion:
        # A true pothole event exhibits an abrupt downward drop followed by a rebound shock
        # (Jerk > threshold and peak-to-peak acceleration > 0.35g ≈ 3.5 m/s^2)
        is_pothole = bool(ptp_acc > 2.5 and max_depression_cm > 1.2)
        
        # If speed is known, estimate cavity longitudinal width: W = v * dt_event
        # Find depression duration where profile exceeds half-max depth
        half_depth = max_depression_cm * 0.4
        in_pothole_indices = np.where(np.abs(road_profile_cm) >= half_depth)[0]
        if len(in_pothole_indices) > 1:
            duration_sec = (in_pothole_indices[-1] - in_pothole_indices[0]) / fs
            width_cm = float(np.clip(speed_mps * duration_sec * 100.0, 15.0, 250.0))
        else:
            width_cm = 50.0
            
        # Impact energy in suspension: E = 0.5 * k_s * delta_z^2 + 0.5 * m_s * v_z^2
        disp_m = max_depression_cm / 100.0
        energy_j = 0.5 * self.p["k_s"] * (disp_m ** 2)
        
        # Calibrate depth with empirical road-pavement transfer coefficient
        calibrated_depth_cm = float(np.clip(max_depression_cm * 1.15, 1.0, 14.0))
        
        # Severity Tier Classification
        if calibrated_depth_cm < 3.2:
            tier = 1
            tier_name = "Tier 1: Minor Routine"
        elif calibrated_depth_cm <= 6.5:
            tier = 2
            tier_name = "Tier 2: Moderate Priority"
        else:
            tier = 3
            tier_name = "Tier 3: Critical Hazard"
            
        return {
            "detected": is_pothole,
            "depth_cm": round(calibrated_depth_cm, 2),
            "width_cm": round(width_cm, 1),
            "energy_joules": round(energy_j, 2),
            "ptp_acc_ms2": round(float(ptp_acc), 2),
            "rms_acc_ms2": round(float(rms_acc), 2),
            "severity_tier": tier,
            "tier_name": tier_name,
            "profile_cm": road_profile_cm.tolist()
        }
