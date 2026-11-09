# Dataset Architecture & Row-Level Documentation

This document provides a comprehensive structural and row-level breakdown of all dashcam and sensor datasets extracted in [`data/`](file:///d:/DSN3099-EPICS/Implementation/data).

---

## 1. Directory Tree

```
Implementation/
└── data/
    ├── dashcam_data/
    │   ├── Indian Roads Dataset/
    │   │   └── Dataset3Class/               # 4,462 JPG images + 4,462 YOLO .txt labels (1.75 GB)
    │   ├── RDD2022 India Pothole D40/
    │   │   └── train/
    │   │       ├── images/                  # 1,530 JPG images (97.3 MB)
    │   │       └── annotations/xmls/        # 1,530 Pascal VOC XML annotations
    │   └── RDD2022_released_through_CRDDC2022/
    │       └── RDD2022/India/
    │           ├── train/ (images, xmls)    # 7,706 images + 7,706 Pascal VOC XML annotations (625 MB)
    │           └── test/ (images)           # 1,959 test images
    │
    └── sensor_data/
        ├── Intelligent Vehicle Perception/  # PVS 1 to PVS 9 (81 sensor/map CSVs + HTML, 2.36 GB)
        ├── Pothole, Road Quality Detection Sensor data/
        │   ├── Pothole/                     # 5 trips with sensor logs & pothole timestamp labels
        │   └── RoadCondition/               # 15 trips labeled as Good vs Bad road condition
        ├── sensor_data_v2/                  # 4 collector folders (99 CSVs with road surface classes)
        ├── SmartRoadSense OpenData/         # 1 large OpenData CSV (335.5 MB)
        └── Two Wheeler Road Surface Classifier/
            ├── Raw Data/                    # 101 trip CSVs across 4 surface types (136.4 MB)
            └── Preprocessed Data/           # 5,795 windowed sensor segments (3-second windows)
```

---

## 2. High-Level Summary Table

| Category | Dataset | Row-Level Entity | Key Columns | Total Rows across Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor** | [**SmartRoadSense OpenData**](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/SmartRoadSense%20OpenData) | 1 geo-located road roughness score | `latitude`, `longitude`, `ppe`, `osm_id`, `highway` | **3,615,539 rows** |
| **Sensor** | [**Intelligent Vehicle Perception**](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Intelligent%20Vehicle%20Perception) | 1 synchronized 100 Hz IMU time step | `timestamp`, `acc_x/y/z`, `gyro_x/y/z`, `mag_x/y/z`, `temp` | **1,080,905 rows** (IMU + Labels)<br>**11,754 rows** (GPS) |
| **Sensor** | [**Two Wheeler Surface Classifier**](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Two%20Wheeler%20Road%20Surface%20Classifier) | 1 instantaneous two-wheeler IMU reading | `ID`, `SrNo`, `Timestamp`, `X/Y/Z_Acc`, `X/Y/Z_Gyro` | **254,046 raw rows**<br>(5,795 preprocessed 3s window files) |
| **Sensor** | [**sensor_data_v2**](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/sensor_data_v2) | 1 mobile phone accelerometer reading | `timestamp`, `X/Y/Z-raw`, `X/Y/Z-axis`, `road_type` | **42,599 rows** (across 99 CSVs) |
| **Sensor** | [**Pothole & Road Quality**](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Pothole,%20Road%20Quality%20Detection%20Sensor%20data) | 1 trip sensor sample / 1 pothole event | `timestamp`, `lat`, `lon`, `speed`, `accelX/Y/Z`, `gyroX/Y/Z` | **21,188 sensor rows**<br>**96 labeled pothole events** |
| **Vision** | [**Indian Roads Dataset**](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/Indian%20Roads%20Dataset) | 1 bounding box annotation (YOLO format) | `<class_id> <x_center> <y_center> <width> <height>` | **4,462 image-label pairs**<br>(Multiple bbox rows per `.txt` file) |
| **Vision** | [**RDD2022 India Pothole D40**](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/RDD2022%20India%20Pothole%20D40) | 1 pothole (`D40`) bounding box object (VOC) | `<name>`, `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>` | **1,530 images + 1,530 XML files** |
| **Vision** | [**RDD2022 CRDDC2022 India**](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/RDD2022_released_through_CRDDC2022) | 1 road damage bounding box object (VOC) | `<name>` (`D00`-`D44`), `<xmin/ymin/xmax/ymax>` | **7,706 train XMLs + 1,959 test images** |

---

## 3. Sensor & Telemetry Datasets (Tabular CSVs)

### 3.1 SmartRoadSense OpenData

* **Folder Path:** [`data/sensor_data/SmartRoadSense OpenData`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/SmartRoadSense%20OpenData)
* **File:** [`open_data.csv`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/SmartRoadSense%20OpenData/open_data.csv) (335.5 MB)
* **Total Rows:** **3,615,539 rows**
* **What 1 row represents:** An aggregated pavement roughness calculation recorded at a specific GPS coordinate and matched to an OpenStreetMap (OSM) highway segment.

#### Schema
```csv
"latitude","longitude","ppe","osm_id","highway","quality","passengers","updated_at"
```

#### Column Details
* `latitude`, `longitude`: WGS84 geographic coordinate (degrees).
* `ppe`: **Pavement Profile Estimation** (continuous surface roughness index calculated from mobile accelerometers).
* `osm_id`: OpenStreetMap road line segment ID.
* `highway`: Road category (e.g., `trunk`, `primary`, `secondary`, `residential`).
* `quality`, `passengers`: Road quality classification and vehicle occupancy indicator.
* `updated_at`: Timestamp of calculation (`YYYY-MM-DD HH:MM:SS.mmmmmm`).

#### Sample Row
```csv
41.841706933505,15.711609810286,0.0791611754371345,230540718,trunk,,,2016-08-29 01:49:41.969818
```

---

### 3.2 Intelligent Vehicle Perception (PVS 1 to PVS 9)

* **Folder Path:** [`data/sensor_data/Intelligent Vehicle Perception`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Intelligent%20Vehicle%20Perception)
* **Total Rows:** **1,080,905 IMU/Label rows** + **11,754 GPS rows** across 9 trips (`PVS 1` through `PVS 9`).
* **Vehicles & Scenarios:** 3 vehicles (Volkswagen Saveiro, Fiat Bravo, Fiat Palio), 3 drivers, 3 driving routes.
* **Sampling Rate:** IMU sensors at **100 Hz** (~10 ms per row), GPS at **1 Hz**.

#### A. Synchronized Multi-Sensor Row (`dataset_gps_mpu_left.csv` / `dataset_gps_mpu_right.csv`)
* **What 1 row represents:** A single 100 Hz synchronized time step measuring multi-sensor telemetry across 3 mounting locations (Dashboard, Above Suspension, Below Suspension).
* **Schema (32 columns):**
  ```csv
  timestamp,
  acc_x_dashboard, acc_y_dashboard, acc_z_dashboard,
  acc_x_above_suspension, acc_y_above_suspension, acc_z_above_suspension,
  acc_x_below_suspension, acc_y_below_suspension, acc_z_below_suspension,
  gyro_x_dashboard, gyro_y_dashboard, gyro_z_dashboard,
  gyro_x_above_suspension, gyro_y_above_suspension, gyro_z_above_suspension,
  gyro_x_below_suspension, gyro_y_below_suspension, gyro_z_below_suspension,
  mag_x_dashboard, mag_y_dashboard, mag_z_dashboard,
  mag_x_above_suspension, mag_y_above_suspension, mag_z_above_suspension,
  temp_dashboard, temp_above_suspension, temp_below_suspension,
  timestamp_gps, latitude, longitude, speed
  ```
* **Sample Row:**
  ```csv
  1577218796.56,0.365,0.168,9.794,0.328,0.173,9.782,0.025,0.173,9.794,-0.134,-0.019,0.138,0.138,0.160,-0.073,-0.042,0.167,-0.078,22.88,-6.04,1.77,75.34,-7.54,-0.33,35.07,33.34,31.78,1577218795.999,-27.71784,-51.09887,0.009
  ```

#### B. Ground Truth Label Row (`dataset_labels.csv`)
* **What 1 row represents:** One-hot encoded multi-label ground truth strictly aligned 1-to-1 with every single sensor reading row.
* **Schema (14 columns):**
  ```csv
  paved_road,unpaved_road,dirt_road,cobblestone_road,asphalt_road,no_speed_bump,speed_bump_asphalt,speed_bump_cobblestone,good_road_left,regular_road_left,bad_road_left,good_road_right,regular_road_right,bad_road_right
  ```
* **Sample Row:**
  ```csv
  1,0,0,0,1,1,0,0,1,0,0,1,0,0
  ```
  *(Interpretation: Paved asphalt road, no speed bump, good road surface condition on both left and right wheels).*

#### C. GPS Telemetry Row (`dataset_gps.csv`)
* **What 1 row represents:** 1-second interval vehicle navigation telemetry.
* **Schema (20 columns):**
  ```csv
  timestamp,latitude,longitude,elevation,accuracy,bearing,speed_meters_per_second,satellites,provider,hdop,vdop,pdop,geoidheight,ageofdgpsdata,dgpsid,activity,battery,annotation,distance_meters,elapsed_time_seconds
  ```

---

### 3.3 Two Wheeler Road Surface Classifier

* **Folder Path:** [`data/sensor_data/Two Wheeler Road Surface Classifier`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Two%20Wheeler%20Road%20Surface%20Classifier)
* **Total Rows:** **254,046 raw rows** across 99 trip CSVs, plus **5,795 preprocessed window files** (each window file contains ~150 to ~250 rows, representing 3-second segments).
* **Sampling Rate:** ~50 Hz.
* **What 1 row represents:** A single smartphone IMU reading on a two-wheeler (motorcycle/scooter).

#### Schema
```csv
ID,SrNo,Timestamp,X_Acc,Y_Acc,Z_Acc,X_Gyro,Y_Gyro,Z_Gyro
```

#### Column Details
* `ID`: Driver / collector identifier (`Abj`, `Ank`, `Har`).
* `SrNo`: Incremental reading index within the recording session.
* `Timestamp`: System timestamp in milliseconds (Unix epoch).
* `X_Acc`, `Y_Acc`, `Z_Acc`: 3-axis linear acceleration in $\text{m/s}^2$ ($Z$ captures vertical bump shocks).
* `X_Gyro`, `Y_Gyro`, `Z_Gyro`: Angular velocity rates around X, Y, Z axes (pitch, roll, yaw).

#### Sample Row
```csv
Abj,1,1739352171418,-0.9679,-1.29452,9.83719,-7.15616,7.56814,8.25476
```

#### Target Classes (Folder Partitioned)
1. **Bitumin:** Asphalt / Tar road.
2. **Block:** Interlocking paver blocks.
3. **Concrete:** Cement concrete road.
4. **Kanker:** Gravel, broken stone, unpaved dirt.

---

### 3.4 sensor_data_v2

* **Folder Path:** [`data/sensor_data/sensor_data_v2`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/sensor_data_v2)
* **Total Rows:** **42,599 rows** across 99 individual trip files.
* **Collectors:** `RoadSurfaceDataCollector1` (22 files), `RoadSurfaceDataCollector2` (34 files), `RoadSurfaceDataCollector5` (42 files), `samsungs2` (1 file).
* **What 1 row represents:** A single mobile accelerometer reading paired with a high-level road surface condition label.

#### Schema
```csv
# timestamp,X-raw,Y-raw,Z-raw,X-axis,Y-axis,Z-axis,road_type
```

#### Column Details
* `timestamp`: Epoch milliseconds.
* `X-raw`, `Y-raw`, `Z-raw`: Raw uncalibrated acceleration values.
* `X-axis`, `Y-axis`, `Z-axis`: Filtered gravity-corrected acceleration values.
* `road_type`: Categorical label: **`Bumpy`**, **`Rough`**, or **`Smooth`**.

#### Sample Row
```csv
1495513860445,-0.078347,0.776376,-0.949172,-1.124958,0.110602,0.481730,Bumpy
```

---

### 3.5 Pothole & Road Quality Detection Sensor Data

* **Folder Path:** [`data/sensor_data/Pothole, Road Quality Detection Sensor data`](file:///d:/DSN3099-EPICS/Implementation/data/sensor_data/Pothole,%20Road%20Quality%20Detection%20Sensor%20data)
* **Total Rows:** **21,188 sensor rows** across 20 trips + **96 ground-truth pothole timestamps**.

#### A. Sensor Stream Files (`trip*_sensors.csv`, `good*_sensors.csv`, `bad*_sensors.csv`)
* **What 1 row represents:** Instantaneous combined GPS + 6-axis motion reading sampled at ~5-10 Hz.
* **Schema:**
  ```csv
  timestamp,latitude,longitude,speed,accelerometerX,accelerometerY,accelerometerZ,gyroX,gyroY,gyroZ
  ```
* **Sample Row:**
  ```csv
  1492638964.5,40.447444787735,-79.9441886564716,0.0,0.01699,-0.96223,0.20388,-0.01699,0.01925,0.00723
  ```

#### B. Pothole Ground-Truth Files (`trip*_potholes.csv`)
* **What 1 row represents:** The exact Unix timestamp when the vehicle wheel struck a pothole on the road.
* **Schema:**
  ```csv
  timestamp
  ```
* **Sample Row:**
  ```csv
  1492639065.7
  ```

---

## 4. Vision & Dashcam Annotation Datasets

### 4.1 Indian Roads Dataset (Dataset3Class)

* **Folder Path:** [`data/dashcam_data/Indian Roads Dataset/Dataset3Class`](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/Indian%20Roads%20Dataset/Dataset3Class)
* **Format:** Darknet / YOLO `.txt` files matching each `.jpg` image (4,462 image-label pairs).
* **What 1 row represents:** One detected bounding box enclosing a road defect / obstacle.

#### Schema (Space-separated)
```txt
<class_id> <x_center> <y_center> <width> <height>
```
*(All coordinates are float values normalized to `[0.0, 1.0]` relative to image width and height).*

#### Class Mapping
* `0`, `1`, `2`: Multi-class road anomaly labels (potholes, speed bumps, surface damage).

#### Sample Rows from an annotation file
```txt
1 0.394792 0.704504 0.538542 0.464154
1 0.455729 0.176011 0.287500 0.335478
2 0.755469 0.174632 0.226562 0.220588
```

---

### 4.2 RDD2022 India Datasets (Pascal VOC XML)

* **Folder Paths:**
  * Curated Pothole Subset: [`data/dashcam_data/RDD2022 India Pothole D40`](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/RDD2022%20India%20Pothole%20D40) (1,530 images + 1,530 XMLs)
  * Full CRDDC2022 India: [`data/dashcam_data/RDD2022_released_through_CRDDC2022`](file:///d:/DSN3099-EPICS/Implementation/data/dashcam_data/RDD2022_released_through_CRDDC2022) (7,706 train XMLs + 1,959 test images)
* **Format:** Pascal VOC XML annotations.
* **What each `<object>` block represents:** One road damage bounding box instance in pixel coordinates.

#### Object XML Schema
```xml
<object>
    <name>D40</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
        <xmin>20</xmin>
        <ymin>473</ymin>
        <xmax>368</xmax>
        <ymax>576</ymax>
    </bndbox>
</object>
```

#### Road Damage Class Dictionary (`<name>`)
| Code | Damage Category | Description |
| :--- | :--- | :--- |
| `D00` | Longitudinal Crack | Wheel mark / linear longitudinal road crack |
| `D10` | Transverse Crack | Construction joint / lateral crack perpendicular to lane |
| `D20` | Alligator Crack | Fatigue cracking forming interconnected crocodile-skin mesh |
| `D40` | **Pothole** | Structural bowl-shaped cavity/depression in the road |
| `D43` | Crosswalk Blur | Degraded/eroded zebra crossing paint lines |
| `D44` | White Line Blur | Faded/worn lane divider markings |
