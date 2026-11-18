"""
=============================================================================
STEP 2: MODEL TRAINING (YOLOv8s ON EXPANDED 5,600+ POTHOLES DATASET)
=============================================================================
Author: EPICS DSN3099
Focus:
  - Deep Learning Fine-Tuning on 2,515 Training Images (4,725 Potholes)
  - Multi-scale mosaic augmentations for clustered/multi-pothole scenarios
  - Device: NVIDIA GeForce RTX 4060 GPU
=============================================================================
"""

import os
import sys
import shutil

def train_model(epochs=25, batch_size=16, imgsz=640, model_type="yolov8s.pt"):
    from ultralytics import YOLO
    import torch

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_yaml = os.path.join(project_root, "data", "pothole_detector_dataset", "pothole_dataset.yaml")
    output_checkpoints = os.path.join(project_root, "pothole_detection_pipeline", "checkpoints")
    os.makedirs(output_checkpoints, exist_ok=True)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print("\n" + "="*70)
    print("STEP 2: TRAINING HIGH-ACCURACY POTHOLE DETECTOR ON RTX 4060 GPU")
    print("="*70)
    print(f"Architecture       : {model_type}")
    print(f"Dataset YAML       : {data_yaml}")
    print(f"Hardware Device    : {device} ({device_name})")
    print(f"Hyperparameters    : Epochs={epochs}, BatchSize={batch_size}, ImageSize={imgsz}x{imgsz}")
    print(f"Optimizer          : AdamW (Cosine Annealing LR)")
    
    model = YOLO(model_type)
    
    training_run = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=output_checkpoints,
        name="training_run_expanded",
        exist_ok=True,
        save=True,
        plots=True,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,         # Multi-pothole cluster learning
        mixup=0.15,         # Road surface texture variance
        scale=0.5,          # Multi-scale pyramid for distant / small cavities
        close_mosaic=5,     # Crisp bounding box boundary refinement in final epochs
        hsv_h=0.015,
        hsv_s=0.6,          # Asphalt shine and wet/dry road variance
        hsv_v=0.4,          # Tree shadows and harsh sunlight variance
        box=7.5,            # High penalty on box coordinate error
        cls=0.5,
        dfl=1.5,
        patience=10
    )
    
    run_best = os.path.join(output_checkpoints, "training_run_expanded", "weights", "best.pt")
    final_best = os.path.join(output_checkpoints, "best_pothole_model.pt")
    
    if os.path.exists(run_best):
        shutil.copy2(run_best, final_best)
        print(f"\n[SUCCESS] Best trained checkpoint copied to: {final_best}")
    else:
        print(f"\n[WARNING] Checkpoint found at {run_best}")
        final_best = run_best
        
    print("="*70 + "\n")
    return final_best

if __name__ == "__main__":
    train_model(epochs=25)
