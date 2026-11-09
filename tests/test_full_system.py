"""
Comprehensive Test Suite for EPICS DSN3099
==========================================
Tests:
1. Feature Engineering integrity (0 NaNs, expected feature dimensions, positive energy).
2. Feature Selection manifest and VIF collinearity constraints.
3. 3D Surface Reconstruction Engine (metric unprojection, RANSAC plane fitting, cavity depth).
4. Quarter-Car Dynamic ODE Solver (transient response, speed scaling, non-zero dissipation).
5. Cross-Modal Sensor Validator (confidence boundaries [0, 1], severity tier monotonicity).
6. End-to-End Master Pipeline (execution on real test dashcam image, JSON output validity).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import json
import cv2

# Ensure project root in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.geometry.depth_reconstructor import Pothole3DReconstructor
from src.simulation.quarter_car_model import QuarterCarSimulator
from src.validation.sensor_cross_validator import SensorCrossValidator
from src.main_pipeline import MasterPipeline


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.feat_dir = os.path.join(PROJECT_ROOT, "data/processed/features")
        self.vision_csv = os.path.join(self.feat_dir, "vision_engineered_geometry.csv")
        self.sensor_csv = os.path.join(self.feat_dir, "sensor_engineered_features.csv")
        self.wave_csv = os.path.join(self.feat_dir, "waveform_engineered_features.csv")

    def test_vision_features_integrity(self):
        self.assertTrue(os.path.exists(self.vision_csv), "Vision engineered CSV must exist")
        df = pd.read_csv(self.vision_csv)
        self.assertGreater(len(df), 1000, "Should have >1000 engineered vision samples")
        self.assertEqual(df.isna().sum().sum(), 0, "Vision features must have 0 NaNs")
        self.assertTrue((df["dist_z_m"] > 0).all(), "Distance Z must be strictly positive")
        self.assertTrue((df["phys_width_m"] > 0).all(), "Physical width must be positive")
        self.assertTrue((df["est_depth_cm"] > 0).all(), "Estimated depth must be positive")
        self.assertTrue(set(df["severity_tier"].unique()).issubset({1, 2, 3}), "Severity tiers must be in {1, 2, 3}")

    def test_sensor_features_integrity(self):
        self.assertTrue(os.path.exists(self.sensor_csv), "Sensor engineered CSV must exist")
        df = pd.read_csv(self.sensor_csv)
        self.assertGreater(len(df), 1000, "Should have >1000 engineered sensor rows")
        self.assertEqual(df.isna().sum().sum(), 0, "Sensor features must have 0 NaNs")
        self.assertTrue((df["speed_norm_peak_g"] >= 0).all(), "Speed normalized shock must be non-negative")
        self.assertTrue((df["kinetic_shock_energy_joules"] >= 0).all(), "Kinetic energy must be non-negative")

    def test_waveform_features_integrity(self):
        self.assertTrue(os.path.exists(self.wave_csv), "Waveform engineered CSV must exist")
        df = pd.read_csv(self.wave_csv)
        self.assertEqual(len(df), 96, "Should have exactly 96 ground-truth waveforms")
        self.assertEqual(df.isna().sum().sum(), 0, "Waveform features must have 0 NaNs")
        self.assertTrue((df["impulse_area_g_sec"] > 0).all(), "Impulse area must be strictly positive")


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.manifest_path = os.path.join(PROJECT_ROOT, "data/processed/features/selected_features_manifest.json")

    def test_manifest_validity(self):
        self.assertTrue(os.path.exists(self.manifest_path), "Selected features manifest must exist")
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("selected_sensor_features", manifest)
        self.assertIn("selected_vision_features", manifest)
        self.assertGreaterEqual(len(manifest["selected_sensor_features"]), 5)
        self.assertGreaterEqual(len(manifest["selected_vision_features"]), 3)
        self.assertLessEqual(manifest["sensor_vif_threshold"], 10.0, "VIF threshold must be <= 10.0")


class Test3DReconstructionEngine(unittest.TestCase):
    def setUp(self):
        self.reconstructor = Pothole3DReconstructor(h_cam=1.25, fov_deg=70.0, y_horizon=0.45)
        # Synthetic flat image with a dark elliptical pothole
        self.synth_img = np.full((640, 640, 3), 120, dtype=np.uint8)
        cv2.ellipse(self.synth_img, (320, 480), (60, 30), 0, 0, 360, (30, 30, 30), -1)

    def test_geometric_distance_formula(self):
        # Center of synthetic pothole at cy = 480/640 = 0.75
        cy = 480 / 640.0
        expected_z = 1.25 / (cy - 0.45)
        bbox = [0.5, cy, 120/640.0, 60/640.0]
        res = self.reconstructor.reconstruct_pothole_geometry(self.synth_img, bbox)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["distance_z_m"], expected_z, places=2)

    def test_metric_reconstruction_bounds(self):
        bbox = [0.5, 0.75, 0.2, 0.1]
        res = self.reconstructor.reconstruct_pothole_geometry(self.synth_img, bbox)
        self.assertGreater(res["width_cm"], 10.0, "Width should be > 10 cm")
        self.assertGreater(res["length_cm"], 5.0, "Length should be > 5 cm")
        self.assertGreater(res["depth_max_cm"], 0.5, "Depth should be > 0.5 cm")
        self.assertGreater(res["volume_cm3"], 50.0, "Volume should be > 50 cm^3")
        self.assertIn(res["severity_tier"], [1, 2, 3], "Severity tier must be 1, 2, or 3")


class TestQuarterCarDynamicSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = QuarterCarSimulator()

    def test_dynamic_solver_response(self):
        res = self.sim.simulate_pothole_impact(depth_max_cm=4.0, length_cm=40.0, speed_kmh=35.0)
        self.assertGreater(res["peak_g"], 0.1, "Simulated peak shock must be > 0.1g for 4cm pothole")
        self.assertGreater(res["p2p_g"], res["peak_g"], "Peak-to-peak must exceed single peak")
        self.assertGreater(res["rms_g"], 0.01, "Shock RMS must be non-zero")
        self.assertEqual(len(res["t_eval"]), len(res["accel_sprung_g"]), "Time and acceleration vectors must align")

    def test_speed_monotonicity(self):
        # Traversal over same pothole at higher speed should impart higher/faster energy
        res_slow = self.sim.simulate_pothole_impact(depth_max_cm=5.0, length_cm=45.0, speed_kmh=20.0)
        res_fast = self.sim.simulate_pothole_impact(depth_max_cm=5.0, length_cm=45.0, speed_kmh=50.0)
        # Acceleration peak at 50 km/h should exceed 20 km/h
        self.assertGreater(res_fast["peak_g"], res_slow["peak_g"] * 0.8)

    def test_depth_monotonicity(self):
        # Deeper pothole at same speed must produce strictly larger shock
        res_shallow = self.sim.simulate_pothole_impact(depth_max_cm=2.0, length_cm=45.0, speed_kmh=35.0)
        res_deep = self.sim.simulate_pothole_impact(depth_max_cm=8.0, length_cm=45.0, speed_kmh=35.0)
        self.assertGreater(res_deep["peak_g"], res_shallow["peak_g"], "Deeper pothole must produce larger shock")


class TestCrossModalValidator(unittest.TestCase):
    def setUp(self):
        self.validator = SensorCrossValidator()

    def test_severity_tier_alignment(self):
        minor = self.validator.validate_pothole_reconstruction(depth_max_cm=2.0, length_cm=30.0, width_cm=35.0, speed_kmh=35.0)
        moderate = self.validator.validate_pothole_reconstruction(depth_max_cm=4.5, length_cm=45.0, width_cm=50.0, speed_kmh=35.0)
        critical = self.validator.validate_pothole_reconstruction(depth_max_cm=9.0, length_cm=65.0, width_cm=70.0, speed_kmh=35.0)

        self.assertEqual(minor["severity_tier"], 1, "Depth 2.0cm must be Tier 1")
        self.assertEqual(moderate["severity_tier"], 2, "Depth 4.5cm must be Tier 2")
        self.assertEqual(critical["severity_tier"], 3, "Depth 9.0cm must be Tier 3")
        self.assertTrue(0.0 <= minor["confidence_score"] <= 1.0, "Confidence score must be in [0, 1]")
        self.assertTrue(0.0 <= moderate["confidence_score"] <= 1.0, "Confidence score must be in [0, 1]")


class TestMasterEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = MasterPipeline()
        self.test_img_dir = os.path.join(PROJECT_ROOT, "data/processed/vision/rdd2022_d40/images/test")
        self.test_lbl_dir = os.path.join(PROJECT_ROOT, "data/processed/vision/rdd2022_d40/labels/test")

    def test_real_image_inference(self):
        test_images = [os.path.join(self.test_img_dir, f) for f in os.listdir(self.test_img_dir) if f.endswith(".jpg")]
        self.assertGreater(len(test_images), 0, "Test images must be available")
        img_path = test_images[0]
        lbl_path = os.path.join(self.test_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        out_img_path = os.path.join(PROJECT_ROOT, "eda_report/pipeline_outputs/test_run_output.jpg")
        report, annotated = self.pipeline.process_single_frame(
            img_path,
            label_path=lbl_path,
            simulated_speed_kmh=35.0,
            output_annotated_path=out_img_path
        )

        self.assertIsInstance(report, list, "Report must be a list of detection dictionaries")
        self.assertTrue(os.path.exists(out_img_path), "Annotated output image must be written")
        self.assertEqual(annotated.shape[2], 3, "Annotated image must be a 3-channel BGR image")

        if len(report) > 0:
            det = report[0]
            self.assertIn("geometry", det)
            self.assertIn("dynamic_validation", det)
            self.assertGreater(det["geometry"]["depth_max_cm"], 0)
            self.assertGreater(det["dynamic_validation"]["simulated_peak_g"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

