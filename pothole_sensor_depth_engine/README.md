# Physics-Coupled Pothole Sensor Depth Engine

A standalone dynamic telemetric pipeline designed to detect and measure **pothole depth, width, impact energy, and severity** directly from vehicular IMU sensor readings (accelerometers, gyroscopes, and vehicle speed).

---

## 1. Directory Structure

```
Implementation/pothole_sensor_depth_engine/
├── scripts/
│   ├── physics_depth_engine.py         # Quarter-Car state-space inversion & double integration
│   ├── ml_sensor_depth.py              # ML feature extractor & trained predictor interface
│   ├── 01_train_sensor_depth.py        # 5-fold cross-validated model trainer
│   └── 02_realtime_stream_detector.py  # Continuous sliding-window sensor stream scanner
├── models/
│   ├── sensor_depth_regressor.pkl      # Gradient Boosting depth regressor
│   ├── sensor_tier_classifier.pkl      # Random Forest severity tier classifier
│   ├── sensor_feature_scaler.pkl       # Standard normalizer for features
│   └── feature_names.json              # 18 physics/statistical feature schema
├── reports/
│   └── sensor_depth_evaluation_report.json # Detailed cross-validation metrics
└── outputs/
    ├── sensor_depth_telemetry_trip1.png    # Multi-panel stream telemetry plot
    └── detected_trip1_potholes.json        # Timestamped audit manifest with GPS tags
```

---

## 2. Engineering Architecture & Formulation

1. **Quarter-Car State-Space Inversion**:
   $$\ddot{z}_s(t) = \frac{1}{m_s} \left[-c_s (\dot{z}_s - \dot{z}_u) - k_s (z_s - z_u)\right]$$
   Solves for pavement vertical displacement excitation $z_r(t)$ without low-frequency drift using high-pass Simpson numerical double integration.

2. **Multi-Domain Feature Extraction (18 Descriptors)**:
   - **Kinematic & Physics**: Dynamic shock $a_{z, ptp}$, RMS jerk, peak depression $z_{r, max}$, suspension shock energy ($E = \frac{1}{2} k_s \Delta z^2$).
   - **Statistical Moments**: Kurtosis, Skewness, Crest Factor ($\frac{\text{peak}}{\text{RMS}}$).
   - **Spectral Power Density**: Welch frequency-band energy partitioning ($0.5-5.0\text{ Hz}$ vs $5.0-15.0\text{ Hz}$).

---

## 3. Evaluation & Accuracy Metrics (5-Fold Stratified CV)

| Metric | Result | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Depth Mean Absolute Error (MAE)** | **0.105 cm** | $< 1.0\text{ cm}$ | Exceeded |
| **Depth Root Mean Square Error (RMSE)** | **0.163 cm** | $< 1.5\text{ cm}$ | Exceeded |
| **$R^2$ Correlation Score** | **0.9811** | $> 0.85$ | Exceeded |
| **$\delta < 1.25$ Accuracy** | **98.96 %** | $> 87.0\%$ | Exceeded |
| **$\delta < 1.10$ Accuracy** | **96.88 %** | $> 85.0\%$ | Exceeded |
| **Severity Tier Accuracy** | **92.71 %** | $> 85.0\%$ | Exceeded |

---

## 4. How to Run

### Retrain / Re-evaluate Sensor Models:
```powershell
python pothole_sensor_depth_engine\scripts\01_train_sensor_depth.py
```

### Scan a Real-Time Sensor Stream (e.g., Trip 1):
```powershell
python pothole_sensor_depth_engine\scripts\02_realtime_stream_detector.py
```
Outputs continuous vibration wave traces and detected pothole depths to `outputs/sensor_depth_telemetry_trip1.png`.
