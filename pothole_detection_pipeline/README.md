# Dedicated Pothole Detection & Computer Vision Error Evaluation Pipeline

**Project**: EPICS DSN3099 — Autonomous Vision-Based Road Cavity Detection Engine  
**Hardware Accelerated**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.4)

---

## 1. Directory Structure

```text
Implementation/pothole_detection_pipeline/
│
├── run_pipeline.py                 <-- Master runner (executes stages in sequence)
├── README.md                       <-- Pipeline documentation & run instructions
│
├── scripts/
│   ├── 01_prepare_dataset.py       <-- Stage 1: Curates dataset & injects negative samples
│   ├── 02_train_model.py           <-- Stage 2: Trains YOLO on RTX 4060 GPU
│   ├── 03_evaluate_metrics.py      <-- Stage 3: Evaluates Detection, Error & CV Spatial metrics
│   └── 04_visualize_predictions.py <-- Stage 4: Generates annotated visual overlays on test frames
│
├── checkpoints/                    <-- Saved trained weights (best_pothole_model.pt)
├── reports/                        <-- Evaluation reports (pothole_evaluation_report.json)
└── outputs/                        <-- Output visual test frames with HUD overlay
```

---

## 2. How to Run (In Series)

### Option A: One-Click Master Execution (Runs Everything in Series)

From the project directory:

```powershell
cd d:\DSN3099-EPICS\Implementation\pothole_detection_pipeline
python run_pipeline.py
```

Optional flags:
```powershell
# Custom training epochs (e.g. 20 epochs):
python run_pipeline.py --epochs 20

# If dataset is already prepared, skip stage 1:
python run_pipeline.py --skip-data-prep

# Evaluation & visualization only (using existing best checkpoint):
python run_pipeline.py --eval-only
```

---

### Option B: Step-by-Step Manual Execution (Run Each in Series)

#### Step 1: Prepare the Curated Pothole Dataset
Extracts real Indian road potholes (single and multi-pothole clustered cavities) and injects background negative frames (walls, clean pavements, elevated borders) so the model never confuses elevated structures or vehicles for potholes:
```powershell
python scripts\01_prepare_dataset.py
```
*Output*: Generates `data/pothole_detector_dataset/` with train (1,243), val (153), and test (153) splits.

#### Step 2: Train the Deep Learning Model on GPU
Trains the YOLO road cavity model with multi-scale mosaic augmentations for clustered pothole topology on your NVIDIA GeForce RTX 4060:
```powershell
python scripts\02_train_model.py
```
*Output*: Saves `checkpoints/best_pothole_model.pt` and training curves.

#### Step 3: Run Multi-Metric & CV Spatial Error Evaluation
Evaluates test frames across object detection metrics, counting errors, and computer vision geometry distortion:
```powershell
python scripts\03_evaluate_metrics.py
```
*Output*: Saves `reports/pothole_evaluation_report.json` with Precision, Recall, F1, Miss Rate, Counting MAE/RMSE, mIoU, Center Distance Error, and Scale Distortion.

#### Step 4: Generate Visual Overlays & HUD Predictions
Runs inference on test frames, drawing bounding boxes, confidence tags, and camera HUD banners:
```powershell
python scripts\04_visualize_predictions.py
```
*Output*: Saves annotated images into `outputs/` (e.g. `detected_India_000123.jpg`).

---

## 3. Metrics Computed

1. **Standard Detection Metrics**:
   - Precision ($P$)
   - Recall ($R$)
   - F1-Score
2. **Error Rates**:
   - Miss Rate ($FNR = 1 - Recall$)
   - False Discovery Rate ($FDR = 1 - Precision$)
   - False Positives per Image ($FPPI$)
   - False Alarm Rate on Clean Roads ($FAR$)
3. **Multi-Pothole Counting & Density Metrics**:
   - Mean Absolute Error ($MAE$) on pothole count per frame
   - Root Mean Squared Error ($RMSE$)
   - Exact Count Accuracy (%)
   - Within $\pm 1$ Count Accuracy (%)
4. **Computer Vision Spatial Geometry Errors**:
   - Mean Intersection over Union ($mIoU$) on true potholes
   - Center Point Euclidean Distance Error ($pixels$ and $normalized$)
   - Center Point Localization Jitter ($Std$ in pixels)
   - Aspect Ratio / Slenderness Distortion Error ($|AR_{pred} - AR_{gt}|$)
   - Bounding Box Scale / Area Relative Error (%)
