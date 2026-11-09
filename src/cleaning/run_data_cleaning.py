"""
Master Data Cleaning Pipeline Runner
Executes comprehensive sensor data cleaning and vision data cleaning.
Runs audit verification upon completion.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.cleaning.clean_sensors import run_sensor_cleaning
from src.cleaning.clean_vision import run_vision_cleaning
from src.utils.audit_dataset_counts import run_audit

def main():
    print("#" * 70)
    print("      STARTING MULTI-MODAL DATA CLEANING & REMEDIATION PIPELINE")
    print("#" * 70)
    
    t0 = time.time()
    
    # 1. Sensor Cleaning
    print("\n>>> STEP 1: SENSOR DATA CLEANING")
    run_sensor_cleaning()
    
    # 2. Vision Cleaning
    print("\n>>> STEP 2: VISION DATA CLEANING")
    run_vision_cleaning()
    
    # 3. Final Verification Audit
    print("\n>>> STEP 3: POST-CLEANING VERIFICATION AUDIT")
    run_audit()
    
    elapsed = time.time() - t0
    print("\n" + "#" * 70)
    print(f"      ALL DATA CLEANING & AUDIT COMPLETED IN {elapsed:.1f} SECONDS!")
    print("#" * 70)

if __name__ == "__main__":
    main()

