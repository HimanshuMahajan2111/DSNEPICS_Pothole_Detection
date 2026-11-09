"""
SmartRoadSense OpenData Preprocessing Module
Cleans geographic road roughness scores (PPE), filters valid coordinates,
and aggregates roughness distributions across highway road categories.
Output saved to: data/processed/sensors/smart_road_sense/
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_FILE = os.path.join(DATA_DIR, "sensor_data", "SmartRoadSense OpenData", "open_data.csv")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "sensors", "smart_road_sense")

def process_smartroadsense(chunk_size=100000, max_chunks=10):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return
        
    print(f"Processing SmartRoadSense OpenData from: {INPUT_FILE}")
    
    clean_sample = []
    category_stats = {}
    
    chunks_processed = 0
    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, low_memory=False, skipinitialspace=True):
        chunk.columns = chunk.columns.str.strip().str.replace('"', '')
        # Drop rows with missing lat, lon, or ppe
        c_clean = chunk.dropna(subset=['latitude', 'longitude', 'ppe']).copy()
        c_clean['ppe'] = pd.to_numeric(c_clean['ppe'], errors='coerce')
        c_clean = c_clean[c_clean['ppe'] >= 0]
        
        # Take a representative 5% sample for the clean GIS dataset
        sample_chunk = c_clean.sample(frac=0.05, random_state=42)
        clean_sample.append(sample_chunk[['latitude', 'longitude', 'ppe', 'highway', 'quality']])
        
        # Aggregate statistics by highway type
        for hway, grp in c_clean.groupby('highway'):
            hway_str = str(hway)
            if hway_str not in category_stats:
                category_stats[hway_str] = []
            category_stats[hway_str].extend(grp['ppe'].values[:1000].tolist())
            
        chunks_processed += 1
        if chunks_processed >= max_chunks:
            break
            
    df_clean = pd.concat(clean_sample, ignore_index=True)
    clean_path = os.path.join(OUTPUT_BASE, "smart_road_sense_sample.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"Saved {len(df_clean)} sampled GIS roughness rows to: {clean_path}")
    
    # Summary stats
    stats_rows = []
    for hway, vals in category_stats.items():
        if len(vals) >= 20:
            stats_rows.append({
                'highway_type': hway,
                'sample_count': len(vals),
                'ppe_mean': round(float(np.mean(vals)), 4),
                'ppe_std': round(float(np.std(vals)), 4),
                'ppe_p95': round(float(np.percentile(vals, 95)), 4)
            })
            
    df_stats = pd.DataFrame(stats_rows).sort_values('ppe_mean', ascending=False)
    stats_path = os.path.join(OUTPUT_BASE, "highway_roughness_stats.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"Saved highway category roughness stats to: {stats_path}")
    print(df_stats.head())

if __name__ == "__main__":
    process_smartroadsense()

