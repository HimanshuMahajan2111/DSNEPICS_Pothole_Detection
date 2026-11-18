"""
Column & Row Matching Audit: combined + dashcam_data + sensor_data -> processed
"""
import pandas as pd, os, glob

base = r'd:\DSN3099-EPICS\Implementation\data'
proc = r'd:\DSN3099-EPICS\Implementation\data\processed'

print("=== IVP sensor_data -> processed ivp_suspension ===")
ivp_dir = os.path.join(base, 'sensor_data', 'Intelligent Vehicle Perception')
for f in sorted(glob.glob(ivp_dir + '/**/*.csv', recursive=True))[:4]:
    df = pd.read_csv(f, nrows=1)
    print(f"  {os.path.basename(f)}: {list(df.columns)}")
ivp_proc = pd.read_csv(os.path.join(proc, 'sensors', 'ivp_suspension', 'ivp_suspension_features.csv'), nrows=3)
print(f"  -> processed ivp_suspension_features.csv cols ({len(ivp_proc.columns)}): {list(ivp_proc.columns)}")

print("\n=== SmartRoadSense sensor_data -> processed smart_road_sense ===")
srs_files = sorted(glob.glob(os.path.join(base, 'sensor_data', 'SmartRoadSense', '**', '*.csv'), recursive=True))
print(f"  Found {len(srs_files)} SmartRoadSense CSVs")
if srs_files:
    df = pd.read_csv(srs_files[0], nrows=1)
    print(f"  cols: {list(df.columns)}")
srs_proc = pd.read_csv(os.path.join(proc, 'sensors', 'smart_road_sense', 'smart_road_sense_sample.csv'), nrows=3)
print(f"  -> processed smart_road_sense_sample cols ({len(srs_proc.columns)}): {list(srs_proc.columns)}")

print("\n=== sensor_data_v2 / mobile road roughness ===")
v2_files = sorted(glob.glob(os.path.join(base, 'sensor_data', '**', '*.csv'), recursive=True))
for f in v2_files:
    if 'v2' in f.lower() or 'mobile' in f.lower():
        df = pd.read_csv(f, nrows=1)
        print(f"  {os.path.relpath(f, os.path.join(base,'sensor_data'))[:70]}: {list(df.columns)[:8]}")
        break
mob_proc = pd.read_csv(os.path.join(proc, 'sensors', 'sensor_data_v2', 'mobile_road_roughness.csv'), nrows=3)
print(f"  -> processed mobile_road_roughness cols ({len(mob_proc.columns)}): {list(mob_proc.columns)}")

print("\n=== Two Wheeler sensor_data -> processed two_wheeler ===")
tw_base = os.path.join(base, 'sensor_data', 'Two Wheeler Road Surface Classifier', 'Preprocessed Data', 'FInal Data')
tw_files = sorted(glob.glob(tw_base + '/*.csv'))
if tw_files:
    df = pd.read_csv(tw_files[0], nrows=1)
    print(f"  FinalData/{os.path.basename(tw_files[0])}: {list(df.columns)[:10]}")
tw_proc = pd.read_csv(os.path.join(proc, 'sensors', 'two_wheeler', 'two_wheeler_surface_features.csv'), nrows=3)
print(f"  -> processed two_wheeler_surface_features cols ({len(tw_proc.columns)}): {list(tw_proc.columns)}")

print("\n=== DONE ===")
