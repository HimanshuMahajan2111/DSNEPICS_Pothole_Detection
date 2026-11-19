# Pothole 3D Geometric Reconstruction & Visualization Suite

This module is completely decoupled from the depth estimation training pipeline and is dedicated to generating multi-view 3D reconstructions, bathymetry elevation meshes, and exportable point clouds for civil engineering damage audit.

---

## Directory Architecture

```
Implementation/pothole_3d_reconstruction/
├── scripts/
│   └── 01_visualize_3d_reconstruction.py  # Core 3D Back-Projection & Plotting Engine
├── outputs/                               # Multi-panel high-res diagnostic plots (.png)
│   ├── reconstruction_3d_sample_00000.png
│   ├── reconstruction_3d_sample_00292.png
│   └── ...
├── point_clouds/                          # Exportable 3D Point Clouds (.ply)
│   ├── pothole_3d_sample_00000.ply
│   ├── pothole_3d_sample_00292.ply
│   └── ...
└── reconstruction_summary.json            # Machine-readable audit manifest
```

---

## Diagnostic Panels Generated Per Pothole

Each generated reconstruction figure (`reconstruction_3d_sample_XXXXX.png`) features 4 synchronized multi-view panels:
1. **(A) Road Scene Localization**: Original dashboard scene with camera frustum distance $Z$ and bounding box.
2. **(B) High-Resolution Cavity Texture**: Top-down ortho-crop with metric Width $\times$ Length scale annotations.
3. **(C) Metric Pothole Depth Heatmap**: Pixel-level cavity depression $d(u,v)$ in cm with longitudinal centerline slice.
4. **(D) 3D Bathymetry Elevation Mesh**: Metric 3D surface plot depicting cavity depth, material displacement volume (Litres / $\text{cm}^3$), and civil maintenance severity rating.

---

## How to Run

### Reconstruct any number of candidate potholes:
```powershell
python pothole_3d_reconstruction\scripts\01_visualize_3d_reconstruction.py 15
```

### Viewing Point Clouds:
The `.ply` files stored in `point_clouds/` can be directly opened in:
- [CloudCompare](https://www.danielgm.net/cc/)
- [MeshLab](https://www.meshlab.net/)
- Blender or Autodesk AutoCAD for road resurfacing estimates.
