"""
=============================================================================
MODULE 2: MACHINE LEARNING & DEEP FEATURE SENSOR DEPTH ESTIMATOR
=============================================================================
Combines Physics features (Quarter-Car road displacement, RMS, Jerk, Energy)
with statistical/spectral features (Wavelet/FFT energy, kurtosis, crest factor)
to train a calibrated Regressor & Classifier across real IMU trip data.
=============================================================================
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import pickle
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score

from physics_depth_engine import QuarterCarDepthEngine

class SensorMLDepthEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        else:
            self.models_dir = models_dir
            
        self.qc_engine = QuarterCarDepthEngine()
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.feature_names = None
        self._load_models()

    def _load_models(self):
        reg_p = os.path.join(self.models_dir, "sensor_depth_regressor.pkl")
        clf_p = os.path.join(self.models_dir, "sensor_tier_classifier.pkl")
        scl_p = os.path.join(self.models_dir, "sensor_feature_scaler.pkl")
        fnames_p = os.path.join(self.models_dir, "feature_names.json")
        
        if os.path.exists(reg_p) and os.path.exists(clf_p) and os.path.exists(scl_p):
            with open(reg_p, "rb") as f:
                self.regressor = pickle.load(f)
            with open(clf_p, "rb") as f:
                self.classifier = pickle.load(f)
            with open(scl_p, "rb") as f:
                self.scaler = pickle.load(f)
            with open(fnames_p, "r") as f:
                self.feature_names = json.load(f)
            print("[SensorMLDepthEngine] Loaded pre-trained sensor models.")

    def extract_features(self, acc_z, fs, speed_mps=10.0, acc_in_g=True):
        """
        Computes 18 high-fidelity physical, statistical, and spectral descriptors.
        """
        gravity = 9.80665
        acc_ms2 = acc_z * gravity if acc_in_g else acc_z.copy()
        acc_detrend = acc_ms2 - np.mean(acc_ms2)
        
        # Physics model outputs
        qc_res = self.qc_engine.analyze_pothole_window(acc_z, fs, speed_mps, acc_in_g)
        prof = np.array(qc_res["profile_cm"])
        
        # Statistical moments
        ptp_val = float(np.ptp(acc_detrend))
        std_val = float(np.std(acc_detrend))
        rms_val = float(np.sqrt(np.mean(acc_detrend ** 2)))
        kurt_val = float(kurtosis(acc_detrend))
        skew_val = float(skew(acc_detrend))
        crest_factor = float(ptp_val / (rms_val + 1e-6))
        
        # Jerk (derivative of acceleration)
        dt = 1.0 / fs
        jerk = np.diff(acc_detrend) / dt
        rms_jerk = float(np.sqrt(np.mean(jerk ** 2))) if len(jerk) > 0 else 0.0
        max_jerk = float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0
        
        # Spectral energy distribution (0.5-5Hz, 5-15Hz)
        freqs, psd = welch(acc_detrend, fs=fs, nperseg=min(len(acc_detrend), 32))
        low_band = float(np.sum(psd[(freqs >= 0.5) & (freqs < 5.0)]))
        mid_band = float(np.sum(psd[(freqs >= 5.0) & (freqs <= 15.0)]))
        total_pwr = float(np.sum(psd)) + 1e-8
        spectral_ratio = float(low_band / total_pwr)
        
        feats = {
            "qc_depth_cm": qc_res["depth_cm"],
            "qc_width_cm": qc_res["width_cm"],
            "qc_energy_j": qc_res["energy_joules"],
            "acc_ptp": ptp_val,
            "acc_std": std_val,
            "acc_rms": rms_val,
            "acc_kurtosis": kurt_val,
            "acc_skew": skew_val,
            "crest_factor": crest_factor,
            "rms_jerk": rms_jerk,
            "max_jerk": max_jerk,
            "speed_mps": float(speed_mps),
            "spectral_low_pwr": low_band,
            "spectral_mid_pwr": mid_band,
            "spectral_ratio": spectral_ratio,
            "prof_p95_cm": float(np.percentile(np.abs(prof), 95)),
            "prof_std_cm": float(np.std(prof)),
            "prof_kurt": float(kurtosis(prof)) if len(prof) > 4 else 0.0
        }
        return feats

    def predict(self, acc_z, fs, speed_mps=10.0, acc_in_g=True):
        """
        Runs combined physics + ML inference to output metric depth & severity.
        """
        feats = self.extract_features(acc_z, fs, speed_mps, acc_in_g)
        
        if self.regressor is not None and self.scaler is not None:
            raw_vec = [feats.get(k, 0.0) for k in self.feature_names]
            vec = np.nan_to_num(np.array([raw_vec]), nan=0.0, posinf=1e4, neginf=-1e4)
            vec_sc = self.scaler.transform(vec)
            vec_sc = np.nan_to_num(vec_sc, nan=0.0, posinf=1e4, neginf=-1e4)
            pred_depth = float(np.clip(self.regressor.predict(vec_sc)[0], 0.8, 15.0))
            pred_tier = int(self.classifier.predict(vec_sc)[0])
        else:
            # Fallback to physics engine
            pred_depth = feats["qc_depth_cm"]
            if pred_depth < 3.2:
                pred_tier = 1
            elif pred_depth <= 6.5:
                pred_tier = 2
            else:
                pred_tier = 3
                
        tier_names = {1: "Tier 1: Minor Routine", 2: "Tier 2: Moderate Priority", 3: "Tier 3: Critical Hazard"}
        return {
            "predicted_depth_cm": round(pred_depth, 2),
            "predicted_severity_tier": pred_tier,
            "tier_label": tier_names.get(pred_tier, "Unknown"),
            "estimated_width_cm": feats["qc_width_cm"],
            "impact_energy_j": feats["qc_energy_j"],
            "peak_to_peak_acc": feats["acc_ptp"],
            "features": feats
        }
