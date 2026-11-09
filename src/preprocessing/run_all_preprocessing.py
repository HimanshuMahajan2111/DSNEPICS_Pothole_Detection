"""
Master Preprocessing Pipeline Runner
Executes all specialized preprocessing modules across vision and sensor domains.
Maintains each dataset strictly in its own isolated subfolder.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.preprocessing.preprocess_sensor_two_wheeler import process_two_wheeler_dataset
from src.preprocessing.preprocess_sensor_ivp import process_ivp_trips
from src.preprocessing.preprocess_sensor_v2 import process_sensor_v2
from src.preprocessing.preprocess_sensor_smartroadsense import process_smartroadsense
from src.preprocessing.preprocess_vision_bharat import process_bharat_pothole
from src.preprocessing.preprocess_vision_indian_roads import process_indian_roads

def main():
    print("=" * 60)
    print("STARTING COMPLETE SUBFOLDER PREPROCESSING PIPELINE")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Two-Wheeler Sensor Dataset
    print("\n[1/6] Preprocessing Two-Wheeler Sensor Dataset...")
    try:
        process_two_wheeler_dataset()
    except Exception as e:
        print(f"Error in Two-Wheeler processing: {e}")
        
    # 2. IVP Multisensor Dataset
    print("\n[2/6] Preprocessing Intelligent Vehicle Perception (IVP) Suspension Dataset...")
    try:
        process_ivp_trips(pvs_limit=5)
    except Exception as e:
        print(f"Error in IVP processing: {e}")
        
    # 3. sensor_data_v2 Mobile Accelerometer Dataset
    print("\n[3/6] Preprocessing sensor_data_v2 Mobile Accelerometer Dataset...")
    try:
        process_sensor_v2()
    except Exception as e:
        print(f"Error in sensor_data_v2 processing: {e}")
        
    # 4. SmartRoadSense OpenData GIS Roughness
    print("\n[4/6] Preprocessing SmartRoadSense OpenData...")
    try:
        process_smartroadsense(chunk_size=100000, max_chunks=8)
    except Exception as e:
        print(f"Error in SmartRoadSense processing: {e}")
        
    # 5. BharatPotHole Vision Dataset
    print("\n[5/6] Preprocessing BharatPotHole Vision Dataset...")
    try:
        process_bharat_pothole(max_train=1200, max_val=200, max_test=200)
    except Exception as e:
        print(f"Error in BharatPotHole processing: {e}")
        
    # 6. Indian Roads Dataset
    print("\n[6/6] Preprocessing Indian Roads Vision Dataset...")
    try:
        process_indian_roads(max_samples=1000)
    except Exception as e:
        print(f"Error in Indian Roads processing: {e}")
        
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"ALL SUBFOLDER PREPROCESSING FINISHED IN {elapsed:.1f} SECONDS!")
    print("=" * 60)

if __name__ == "__main__":
    main()

