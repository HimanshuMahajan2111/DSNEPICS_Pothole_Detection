"""
=============================================================================
MASTER EXECUTION RUNNER: POTHOLE DETECTION PIPELINE
=============================================================================
Author: EPICS DSN3099

Executes all 4 pipeline stages in series:
  Stage 1: Dataset Curation & Negative Injection (01_prepare_dataset.py)
  Stage 2: Model Training with Road-Cavity Augmentations (02_train_model.py)
  Stage 3: Full Metric & Computer Vision Error Evaluation (03_evaluate_metrics.py)
  Stage 4: Visual Overlay Generation on Test Frames (04_visualize_predictions.py)

Usage:
  python run_pipeline.py               # Runs all 4 stages in series
  python run_pipeline.py --epochs 20   # Specify custom training epochs
  python run_pipeline.py --eval-only   # Run evaluation & visualization using trained model
=============================================================================
"""

import os
import sys
import argparse
import time

# Add scripts directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from importlib import import_module

def main():
    parser = argparse.ArgumentParser(description="Master Pothole Detection Pipeline Runner")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset curation step")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation & visualization")
    parser.add_argument("--samples", type=int, default=15, help="Number of visualization images to generate")
    args = parser.parse_args()

    start_total = time.time()
    print("\n" + "#"*75)
    print("      EPICS DSN3099: VISION POTHOLE DETECTION & EVALUATION PIPELINE")
    print("#"*75)

    # -------------------------------------------------------------
    # STAGE 1: DATASET CURATION
    # -------------------------------------------------------------
    if not args.eval_only and not args.skip_data_prep:
        t0 = time.time()
        print("\n>>> LAUNCHING STAGE 1: Dataset Preparation...")
        prep_module = import_module("01_prepare_dataset")
        prep_module.prepare_dataset()
        print(f">>> STAGE 1 COMPLETED in {time.time() - t0:.1f}s")
    else:
        print("\n>>> Skipping Stage 1 (Dataset preparation already verified).")

    # -------------------------------------------------------------
    # STAGE 2: MODEL TRAINING
    # -------------------------------------------------------------
    if not args.eval_only:
        t0 = time.time()
        print("\n>>> LAUNCHING STAGE 2: Deep Learning Training on GPU...")
        train_module = import_module("02_train_model")
        best_weights = train_module.train_model(epochs=args.epochs, batch_size=args.batch_size)
        print(f">>> STAGE 2 COMPLETED in {time.time() - t0:.1f}s")
    else:
        print("\n>>> Skipping Stage 2 (Evaluation only mode requested).")

    # -------------------------------------------------------------
    # STAGE 3: MULTI-METRIC & CV ERROR EVALUATION
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n>>> LAUNCHING STAGE 3: Full Metric & Spatial Error Evaluation...")
    eval_module = import_module("03_evaluate_metrics")
    metrics = eval_module.run_evaluation()
    print(f">>> STAGE 3 COMPLETED in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------
    # STAGE 4: VISUAL OVERLAY GENERATION
    # -------------------------------------------------------------
    t0 = time.time()
    print("\n>>> LAUNCHING STAGE 4: Visual Overlay Generation on Test Frames...")
    vis_module = import_module("04_visualize_predictions")
    vis_module.run_visual_inference(num_samples=args.samples)
    print(f">>> STAGE 4 COMPLETED in {time.time() - t0:.1f}s")

    print("\n" + "#"*75)
    print(f"      ALL PIPELINE STAGES COMPLETED SUCCESSFULLY in {time.time() - start_total:.1f}s!")
    print("#"*75)
    print("Outputs Summary:")
    print(f"  - Checkpoints: {os.path.join(current_dir, 'checkpoints')}")
    print(f"  - Evaluation : {os.path.join(current_dir, 'reports', 'pothole_evaluation_report.json')}")
    print(f"  - Visuals    : {os.path.join(current_dir, 'outputs')}")
    print("#"*75 + "\n")

if __name__ == "__main__":
    main()
