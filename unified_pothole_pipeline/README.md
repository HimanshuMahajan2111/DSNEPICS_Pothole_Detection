# Unified Multi-Modal Pothole Reconstruction & Damage Assessment Pipeline

This master pipeline integrates all previously decoupled sub-modules into a single, cohesive, end-to-end framework:
1. **YOLOv8 Detection**: Filters out clean asphalt, vehicles, curbs, and elevated walls, localizing true potholes.
2. **Adaptive Color Segmentation**: Multi-spectral (CIE-LAB & HSV) shadow extraction to isolate the inner cavity boundary.
3. **Foundation Monocular Depth**: MiDaS relative inverse depth back-projection with camera frustum planar alignment.
4. **Quarter-Car Dynamic IMU Inversion**: Reconstructs physical pavement shock $a_{z, ptp}$, jerk, and restitution energy.
5. **Multimodal Tri-Branch Fusion**: Fuses vision, color segmentation, and sensor telemetry to compute calibrated metric depth ($d$ in cm) and severity tier without synthetic leakage.
6. **3D Surface Bathymetry & Point Cloud**: Reconstructs the 3D elevation mesh, excavation volume in Litres/$\text{cm}^3$, and exports standard CAD `.ply` files.

---

## 1. Directory Structure

```
Implementation/unified_pothole_pipeline/
├── scripts/
│   └── unified_pipeline.py            # Master inference engine running all 6 stages
├── models/
│   ├── best_pothole_model.pt          # Fine-tuned YOLOv8 pothole detector
│   ├── fusion_depth_regressor.pkl     # Multimodal Gradient Boosting depth regressor
│   ├── fusion_tier_classifier.pkl     # Multimodal Random Forest severity classifier
│   ├── fusion_scaler.pkl              # Feature standardizer
│   └── feature_names.json             # 19 multimodal feature definitions
├── outputs/
│   ├── unified_demo_dashboard_1.png   # 6-panel synchronized diagnostic dashboard
│   └── unified_demo_pothole_1.ply     # 3D CAD/MeshLab point cloud
└── reports/
    └── unified_demo_audit_report.json # Comprehensive JSON damage assessment
```

---

## 2. 6-Panel Diagnostic Dashboard

Every analyzed pothole generates a single unified visualization (`outputs/*_dashboard_X.png`) comprising:
- **Panel 1**: YOLOv8 road cavity bounding box with confidence score and camera distance $Z$ (meters).
- **Panel 2**: Adaptive CIE-LAB color segmentation overlay depicting inner cavity mask and boundary contours.
- **Panel 3**: Quarter-Car vehicular vertical acceleration $a_z(t)$ wave and dynamic impact shock marker.
- **Panel 4**: Calibrated metric depth depression heatmap ($d$ in cm) with continuous scale.
- **Panel 5**: 3D surface bathymetry elevation mesh showing material displacement volume ($V$ in Litres).
- **Panel 6**: Executive civil engineering maintenance audit card with repair urgency classification.

---

## 3. How to Run

### Run on Any Road Scene Image:
```powershell
python unified_pothole_pipeline\scripts\unified_pipeline.py
```

### Python API Usage:
```python
from unified_pothole_pipeline.scripts.unified_pipeline import UnifiedPotholePipeline

pipeline = UnifiedPotholePipeline()
# Run on an image with optional sensor telemetry:
results = pipeline.run_scene_inference(
    image_path="path/to/road_image.jpg",
    sensor_acc_z=acc_array,   # Optional: pass numpy array of accelerometer Z (g)
    speed_mps=12.0,           # Vehicle speed in m/s (43.2 km/h)
    output_prefix="road_test_01"
)
```
