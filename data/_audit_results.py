"""
Fast Results Audit: processed outputs only (no raw-row counting)
"""
import pandas as pd
import numpy as np
import os, glob

base = r'd:\DSN3099-EPICS\Implementation\data'
proc = r'd:\DSN3099-EPICS\Implementation\data\processed'
sep = "=" * 65

# 1. roadsens_anomaly_features
print(sep)
print("1. combined/RoadSens-4M  ->  roadsens_anomaly_features.csv")
print(sep)
n_src = len(glob.glob(base + r'\combined\RoadSens-4M\**\*.csv', recursive=True))
df = pd.read_csv(os.path.join(proc, 'sensors', 'roadsens_4m', 'roadsens_anomaly_features.csv'))
print(f"  Source files (combined) : {n_src} CSVs")
print(f"  Processed rows          : {len(df):,}")
print(f"  Processed cols ({len(df.columns)})       : {list(df.columns)}")
print(f"  Label dist              : {df['is_anomaly'].value_counts().sort_index().to_dict()}  (0=normal, 1=anomaly)")
print(f"  NaNs                    : {df.isna().sum().sum()}")
print(f"  accel_rms  [{df['accel_rms'].min():.4f} – {df['accel_rms'].max():.4f}]  peak [{df['accel_peak'].min():.4f} – {df['accel_peak'].max():.4f}]")

# 2. sensor_features_pothole
print(f"\n{sep}")
print("2. sensor_data/Pothole trips  ->  sensor_features_pothole.csv")
print(sep)
n_trips = len(glob.glob(os.path.join(base, 'sensor_data', 'Pothole, Road Quality Detection Sensor data', 'Pothole', '*_sensors.csv')))
df2 = pd.read_csv(os.path.join(proc, 'sensors', 'pothole_trips', 'sensor_features_pothole.csv'))
print(f"  Source trip files       : {n_trips} sensor CSVs")
print(f"  Processed rows          : {len(df2):,}  (2-sec windows, 50% overlap)")
print(f"  Processed cols ({len(df2.columns)})      : {list(df2.columns)}")
print(f"  Label dist              : {df2['label'].value_counts().sort_index().to_dict()}  (0=clean, 1=pothole)")
print(f"  NaNs                    : {df2.isna().sum().sum()}")
print(f"  Trips                   : {df2['trip_id'].nunique()} unique")
print(f"  accel_peak range        : [{df2['accel_peak'].min():.4f} – {df2['accel_peak'].max():.4f}] g")
print(f"  mean_speed range        : [{df2['mean_speed'].min():.3f} – {df2['mean_speed'].max():.3f}] m/s")

# 3. pothole_shock_signatures
print(f"\n{sep}")
print("3. sensor_data/Pothole trips  ->  pothole_shock_signatures.csv")
print(sep)
df3 = pd.read_csv(os.path.join(proc, 'sensors', 'pothole_trips', 'pothole_shock_signatures.csv'))
print(f"  Ground-truth potholes   : {len(df3)} events")
print(f"  Cols ({len(df3.columns)})               : {list(df3.columns)}")
print(f"  peak_g range            : [{df3['peak_g'].min():.4f} – {df3['peak_g'].max():.4f}] g")
print(f"  p2p_g  range            : [{df3['p2p_g'].min():.4f} – {df3['p2p_g'].max():.4f}] g")
print(f"  Trips covered           : {df3['trip_id'].nunique()}")

# 4. ivp_suspension
print(f"\n{sep}")
print("4. sensor_data/IVP  ->  ivp_suspension_features.csv")
print(sep)
n_ivp = len(glob.glob(os.path.join(base, 'sensor_data', 'Intelligent Vehicle Perception', '**', '*.csv'), recursive=True))
df4 = pd.read_csv(os.path.join(proc, 'sensors', 'ivp_suspension', 'ivp_suspension_features.csv'))
print(f"  Source IVP files        : {n_ivp} CSVs")
print(f"  Processed rows          : {len(df4):,}")
print(f"  Processed cols ({len(df4.columns)})      : {list(df4.columns)}")
print(f"  NaNs                    : {df4.isna().sum().sum()}")
for c in ['is_paved','is_asphalt','is_speed_bump','is_bad_road']:
    print(f"  {c:25s}: {df4[c].value_counts().sort_index().to_dict()}")

# 5. mobile_road_roughness
print(f"\n{sep}")
print("5. sensor_data/sensor_data_v2  ->  mobile_road_roughness.csv")
print(sep)
n_v2 = len(glob.glob(os.path.join(base, 'sensor_data', 'sensor_data_v2', '**', '*.csv'), recursive=True))
df5 = pd.read_csv(os.path.join(proc, 'sensors', 'sensor_data_v2', 'mobile_road_roughness.csv'))
print(f"  Source v2 files         : {n_v2} CSVs")
print(f"  Processed rows          : {len(df5):,}")
print(f"  Processed cols ({len(df5.columns)})       : {list(df5.columns)}")
print(f"  Road type dist          : {df5['road_type'].value_counts().to_dict()}")
print(f"  NaNs                    : {df5.isna().sum().sum()}")

# 6. two_wheeler
print(f"\n{sep}")
print("6. sensor_data/Two Wheeler  ->  two_wheeler_surface_features.csv")
print(sep)
n_tw = len(glob.glob(os.path.join(base, 'sensor_data', 'Two Wheeler Road Surface Classifier', 'Raw Data', '**', '*.csv'), recursive=True))
df6 = pd.read_csv(os.path.join(proc, 'sensors', 'two_wheeler', 'two_wheeler_surface_features.csv'))
print(f"  Source raw files        : {n_tw} CSVs")
print(f"  Processed rows          : {len(df6):,}")
print(f"  Processed cols ({len(df6.columns)})      : {list(df6.columns)}")
print(f"  Surface class dist      : {df6['surface_class'].value_counts().to_dict()}")
print(f"  NaNs                    : {df6.isna().sum().sum()}")

# 7. dashcam -> vision
print(f"\n{sep}")
print("7. dashcam_data  ->  processed/vision/")
print(sep)
n_raw_jpg = len(glob.glob(os.path.join(base, 'dashcam_data', '**', '*.jpg'), recursive=True))
n_raw_xml = len(glob.glob(os.path.join(base, 'dashcam_data', '**', '*.xml'), recursive=True))
n_raw_txt = len(glob.glob(os.path.join(base, 'dashcam_data', '**', '*.txt'), recursive=True))
proc_imgs = glob.glob(os.path.join(proc, 'vision', '**', '*.jpg'), recursive=True)
proc_lbls = glob.glob(os.path.join(proc, 'vision', '**', '*.txt'), recursive=True)
print(f"  Raw images (.jpg)       : {n_raw_jpg:,}")
print(f"  Raw VOC annotations     : {n_raw_xml:,}")
print(f"  Raw YOLO labels (.txt)  : {n_raw_txt:,}")
print(f"  Processed images        : {len(proc_imgs):,}  (CLAHE + 640×640 letterbox)")
print(f"  Processed label files   : {len(proc_lbls):,}")
# Count boxes quickly
total_boxes = sum(
    sum(1 for l in open(f) if l.strip())
    for f in proc_lbls
)
print(f"  Total pothole bbox rows : {total_boxes:,}")

# split breakdown
for split in ['train','val','test']:
    n = len(glob.glob(os.path.join(proc, 'vision', 'rdd2022_d40', 'images', split, '*.jpg')))
    print(f"    rdd2022_d40 {split:5s}  : {n:,} images")

# 8. feature engineering
print(f"\n{sep}")
print("8. processed/  ->  processed/features/  (feature engineering)")
print(sep)
df_sf = pd.read_csv(os.path.join(proc, 'features', 'sensor_engineered_features.csv'))
df_vf = pd.read_csv(os.path.join(proc, 'features', 'vision_engineered_geometry.csv'))
df_wf = pd.read_csv(os.path.join(proc, 'features', 'waveform_engineered_features.csv'))

print(f"  sensor_engineered_features  : {len(df_sf):,} rows × {len(df_sf.columns)} cols")
print(f"    severity_tier             : {df_sf['severity_tier'].value_counts().sort_index().to_dict()}  (1=minor,2=moderate,3=critical)")
print(f"    kinetic_shock_energy range: [{df_sf['kinetic_shock_energy_joules'].min():.1f} – {df_sf['kinetic_shock_energy_joules'].max():.1f}] J")
print(f"    angular_instability range : [{df_sf['angular_instability'].min():.4f} – {df_sf['angular_instability'].max():.4f}]")

print(f"\n  vision_engineered_geometry  : {len(df_vf):,} rows × {len(df_vf.columns)} cols")
print(f"    severity_tier             : {df_vf['severity_tier'].value_counts().sort_index().to_dict()}")
print(f"    est_depth_cm range        : [{df_vf['est_depth_cm'].min():.2f} – {df_vf['est_depth_cm'].max():.2f}] cm")
print(f"    dist_z_m  range           : [{df_vf['dist_z_m'].min():.2f} – {df_vf['dist_z_m'].max():.2f}] m")
print(f"    volume_cm3 range          : [{df_vf['volume_cm3'].min():.1f} – {df_vf['volume_cm3'].max():.1f}] cm³")

print(f"\n  waveform_engineered_features: {len(df_wf):,} rows × {len(df_wf.columns)} cols")
print(f"    severity_tier             : {df_wf['severity_tier'].value_counts().sort_index().to_dict()}")
print(f"    peak_g range              : [{df_wf['peak_g'].min():.4f} – {df_wf['peak_g'].max():.4f}] g")
print(f"    empirical_zeta range      : [{df_wf['empirical_zeta'].min():.4f} – {df_wf['empirical_zeta'].max():.4f}]")

print(f"\n{sep}")
print("GRAND TOTALS")
print(sep)
sensor_rows = len(df)+len(df2)+len(df3)+len(df4)+len(df5)+len(df6)
feat_rows   = len(df_sf)+len(df_vf)+len(df_wf)
print(f"  Sensor processed rows   : {sensor_rows:,}")
print(f"  Feature engineered rows : {feat_rows:,}")
print(f"  Vision images           : {len(proc_imgs):,}")
print(f"  Pothole bbox rows       : {total_boxes:,}")
print(f"  Total processed records : {sensor_rows + feat_rows:,}  (+{len(proc_imgs):,} images)")
