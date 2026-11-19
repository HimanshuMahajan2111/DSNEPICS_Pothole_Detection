# Skeuobrutalism Pothole Audit & 3D Reconstruction Web Suite

A tactical, high-contrast **Skeuobrutalist Streamlit Web Dashboard** directly wired to the backend multi-modal fusion, color segmentation, and 3D reconstruction pipelines.

---

## 1. Visual & Architectural Design System (Skeuobrutalism)

- **Palette**: Canvas `#f4f0eb`, tactical accent yellow `#ffdf00`, hazard orange `#ff6b35`, neon cyan `#00f0ff`, and signal green `#00e575`.
- **Card Containers**: Solid `3px solid #121212` borders with `5px 5px 0px #121212` hard drop-shadows and tactile elevation shifts on `:hover` and `:active`.
- **Typography**: Industrial headline styling via `Space Grotesk` and terminal code fonts via `JetBrains Mono`.

---

## 2. Integrated Dashboard Features

### Tab 1: 3D Multimodal Analyzer
- **Scene Selector / Live Feed**: Choose real dashcam test images or upload custom images.
- **Dynamic Vehicle Sliders**: Real-time speed adjustment ($15-80\text{ km/h}$) and impact intensity ($g$).
- **Tactile KPI Tiles**: Live metric depth ($d\text{ in cm}$), surface width ($W\text{ in cm}$), repair volume ($V\text{ in Litres}$), and peak shock ($a_z\text{ in }g$).
- **Status Cards**: High-visibility civil maintenance alert boxes (Tier 1 Minor Routine, Tier 2 Moderate Priority, Tier 3 Critical Hazard).
- **Interactive Plotly 3D WebGL Bathymetry**: Real-time rotating elevation mesh with depth contour colorbars.
- **One-Click CAD Export**: Direct browser download for `.ply` 3D point clouds compatible with MeshLab and AutoCAD.
- **Dynamic Waveform Plot**: Acceleration shock curve with impact marker and ground-truth boundary lines.

### Tab 2: Color Segmentation & Vision
- **6-Panel Diagnostic Dashboard**: Synchronization of detection bounding boxes, CIE-LAB color masks, and depth maps.
- **Morphological Metrics**: Cavity fill ratio, shadow contrast ratio, and contour circularity.

### Tab 3: Performance Audit & Telemetry
- **Holdout Test Metrics**: True un-overfitted generalization benchmarks (91.10% Severity Accuracy, 0.17 cm Depth MAE, 99.06% $\delta < 1.25$).
- **Interactive 3×3 Confusion Matrix**: Displays actual vs. predicted classifications across 427 holdout test frames.

---

## 3. How to Launch the Web App

From the workspace root:
```powershell
streamlit run streamlit_app\app.py
```
Or open in your browser:
👉 **`http://localhost:8501`**
