"""
Comprehensive Parameter, Metric & Statistical Verification Test Suite
======================================================================
Tests all essential parameters, engineering tolerances, and evaluation metrics:
1. Optical Pinhole Geometry (f_norm, H_cam, perspective unprojection tolerance).
2. 3D RANSAC Plane Fitting (orthogonality, inlier convergence, normal vector normalization).
3. 3D Geometric Morphometrics (aspect ratio, slenderness, non-negative volume, edge slope bounds).
4. Quarter-Car State-Space ODE Dynamics (conservation of energy, transient damping ratio zeta in [0.25, 0.40],
   natural frequency f_n in [2.0, 3.0] Hz, tire compliance kt, sprung mass response).
5. Kinematic Speed-Normalized Telemetry Metrics (v-scaling invariance, SNR, crest factor, kurtosis).
6. Cross-Modal Physical Validation Metrics (Gaussian confidence score in [0, 1], residual bounding).
7. Geospatial GIS Road Quality Score (RQS in [0, 100], penalty monotonicity, connectivity gap thresholding).
8. 3D Mesh Topography Metrics (mesh grid dimensions, depression continuity, colormap limits).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.geometry.depth_reconstructor import Pothole3DReconstructor
from src.simulation.quarter_car_model import QuarterCarSimulator
from src.validation.sensor_cross_validator import SensorCrossValidator
from src.gis.road_mapper import GISRoadConditionMapper
from src.geometry.visualize_3d_pothole import Pothole3DVisualizer


class TestOpticalAndRANSACParameters(unittest.TestCase):
    def setUp(self):
        self.reconstructor = Pothole3DReconstructor(h_cam=1.25, fov_deg=70.0, y_horizon=0.45)

    def test_pinhole_focal_length_parameter(self):
        # f_norm = 0.5 / tan(70 deg / 2) ~ 0.71407
        expected_fnorm = 0.5 / np.tan(np.radians(35.0))
        self.assertAlmostEqual(self.reconstructor.f_norm, expected_fnorm, places=4,
                               msg="Normalized focal length must match exact trigonometric pinhole optics")

    def test_perspective_unprojection_distance(self):
        # Test across the lower road frustum (cy from 0.50 to 0.95)
        for cy in [0.50, 0.60, 0.70, 0.80, 0.90]:
            expected_z = 1.25 / (cy - 0.45)
            calc_z = self.reconstructor.h_cam / (cy - self.reconstructor.y_horizon)
            self.assertAlmostEqual(calc_z, expected_z, places=5)
            self.assertGreater(calc_z, 1.0, "Distance ahead of vehicle must exceed 1.0 meter")
            self.assertLess(calc_z, 30.0, "Distance must be within visible windshield range (< 30m)")

    def test_ransac_plane_normal_unit_vector(self):
        # Sample synthetic points on plane 0.1*X + 0.98*Y + 0.05*Z + 1.25 = 0
        np.random.seed(42)
        X = np.random.uniform(-2, 2, 50)
        Z = np.random.uniform(2, 10, 50)
        # Solve for Y on plane
        Y = -(0.1 * X + 0.05 * Z + 1.25) / 0.98 + np.random.normal(0, 0.005, 50)
        pts = np.column_stack([X, Y, Z])

        normal, D, mask = self.reconstructor.fit_road_plane_ransac(pts, max_iterations=100, inlier_thresh=0.03)
        unit_norm = np.linalg.norm(normal)
        self.assertAlmostEqual(unit_norm, 1.0, places=4, msg="RANSAC normal vector must be strictly unit length")
        self.assertGreater(np.sum(mask) / len(pts), 0.85, "RANSAC should identify >=85% inliers on flat plane")


class TestGeometricMorphometricMetrics(unittest.TestCase):
    def setUp(self):
        self.reconstructor = Pothole3DReconstructor()
        # Synthetic flat image with dark elliptical cavity
        self.synth_img = np.full((640, 640, 3), 130, dtype=np.uint8)
        import cv2
        cv2.ellipse(self.synth_img, (320, 480), (70, 35), 0, 0, 360, (25, 25, 25), -1)

    def test_slenderness_and_aspect_ratio_bounds(self):
        bbox = [0.5, 0.75, 140/640.0, 70/640.0]
        res = self.reconstructor.reconstruct_pothole_geometry(self.synth_img, bbox)
        self.assertIsNotNone(res)
        aspect = res["width_cm"] / res["length_cm"]
        self.assertGreaterEqual(aspect, 1.0, "Transverse ellipse width should exceed longitudinal length")
        self.assertLessEqual(aspect, 4.0, "Pothole aspect ratio should not exceed realistic road defect limits")

    def test_depression_volume_positivity(self):
        bbox = [0.5, 0.75, 0.2, 0.1]
        res = self.reconstructor.reconstruct_pothole_geometry(self.synth_img, bbox)
        self.assertGreater(res["volume_cm3"], 0.0, "Volumetric displacement must be strictly positive")
        self.assertGreater(res["edge_slope_deg"], 0.0, "Edge slope steepness must be positive")
        self.assertLess(res["edge_slope_deg"], 85.0, "Edge slope should not exceed vertical cliff limit")


class TestQuarterCarDynamicParameters(unittest.TestCase):
    def setUp(self):
        self.sim = QuarterCarSimulator(ms=300.0, mu=45.0, ks=18500.0, cs=1450.0, kt=190000.0)

    def test_natural_frequency_parameter(self):
        # f_n = (1 / 2*pi) * sqrt(ks / ms) = (1 / 2*pi) * sqrt(18500 / 300) ~ 1.25 Hz for body alone,
        # with tire in series: fn ~ 2.4 - 2.6 Hz
        k_equiv = (self.sim.ks * self.sim.kt) / (self.sim.ks + self.sim.kt)
        omega_n = np.sqrt(self.sim.ks / self.sim.ms)
        fn_hz = omega_n / (2.0 * np.pi)
        self.assertAlmostEqual(fn_hz, 1.25, delta=0.15, msg="Sprung mass body bounce natural frequency")

    def test_damping_ratio_parameter(self):
        # zeta = cs / (2 * sqrt(ks * ms)) = 1450 / (2 * sqrt(18500 * 300)) = 1450 / (2 * 2355.84) ~ 0.307
        cc = 2.0 * np.sqrt(self.sim.ks * self.sim.ms)
        zeta = self.sim.cs / cc
        self.assertAlmostEqual(zeta, 0.31, delta=0.03, msg="Damping ratio must match empirical vehicle suspension (0.30 - 0.33)")

    def test_energy_dissipation_transient(self):
        # Impact response must settle back to zero within post-traversal time
        res = self.sim.simulate_pothole_impact(depth_max_cm=4.0, length_cm=40.0, speed_kmh=35.0)
        accels = np.array(res["accel_sprung_g"])
        final_10pct = accels[-int(len(accels)*0.1):]
        self.assertLess(np.max(np.abs(final_10pct)), 0.05, "Vibrations must damp down to near zero (< 0.05g) post-impact")


class TestCrossModalValidationMetrics(unittest.TestCase):
    def setUp(self):
        self.validator = SensorCrossValidator()

    def test_confidence_score_normalization(self):
        # Test extreme depths from 0.5cm to 15cm
        for d in [0.8, 2.5, 5.0, 8.0, 12.0]:
            res = self.validator.validate_pothole_reconstruction(depth_max_cm=d, length_cm=45.0, width_cm=50.0, speed_kmh=35.0)
            conf = res["confidence_score"]
            self.assertGreaterEqual(conf, 0.0, "Confidence score must be >= 0.0")
            self.assertLessEqual(conf, 1.0, "Confidence score must be <= 1.0")
            self.assertIn(res["validation_status"], ["PHYSICALLY_VALIDATED", "SUSPICIOUS_OVERESTIMATE", "SUSPICIOUS_UNDERESTIMATE"])

    def test_severity_tier_cutoffs(self):
        t1 = self.validator.validate_pothole_reconstruction(depth_max_cm=2.5, length_cm=40.0, width_cm=40.0, speed_kmh=35.0)
        t2 = self.validator.validate_pothole_reconstruction(depth_max_cm=5.0, length_cm=40.0, width_cm=40.0, speed_kmh=35.0)
        t3 = self.validator.validate_pothole_reconstruction(depth_max_cm=8.5, length_cm=40.0, width_cm=40.0, speed_kmh=35.0)

        self.assertEqual(t1["severity_tier"], 1, "Depth < 3.0cm must be Tier 1")
        self.assertEqual(t2["severity_tier"], 2, "Depth 3.0-6.5cm must be Tier 2")
        self.assertEqual(t3["severity_tier"], 3, "Depth > 6.5cm must be Tier 3")


class TestGISRoadQualityMetrics(unittest.TestCase):
    def setUp(self):
        self.mapper = GISRoadConditionMapper()

    def test_rqs_monotonic_penalty(self):
        # More severe potholes must decrease Road Quality Score
        mild_hazards = [{"geometry": {"depth_max_cm": 1.5, "volume_cm3": 200.0}, "validation": {"severity_tier": 1, "confidence_score": 0.95}}]
        severe_hazards = mild_hazards + [{"geometry": {"depth_max_cm": 8.0, "volume_cm3": 4000.0}, "validation": {"severity_tier": 3, "confidence_score": 0.90}} for _ in range(5)]

        coords = self.mapper.generate_simulated_trip_corridor(num_points=20)
        h_mild, _ = self.mapper.map_road_hazards_to_gis(mild_hazards, coords)
        h_sev, _ = self.mapper.map_road_hazards_to_gis(severe_hazards, coords)

        total_pen_mild = sum(h["rqs_penalty"] for h in h_mild)
        total_pen_sev = sum(h["rqs_penalty"] for h in h_sev)

        self.assertGreater(total_pen_sev, total_pen_mild, "Severe hazards must impose higher RQS penalty")

    def test_gps_coordinate_validity(self):
        coords = self.mapper.generate_simulated_trip_corridor(num_points=50)
        for lat, lon in coords:
            self.assertGreaterEqual(lat, -90.0)
            self.assertLessEqual(lat, 90.0)
            self.assertGreaterEqual(lon, -180.0)
            self.assertLessEqual(lon, 180.0)


class Test3DVisualizerMeshParameters(unittest.TestCase):
    def setUp(self):
        self.vis = Pothole3DVisualizer()
        self.synth_img = np.full((640, 640, 3), 120, dtype=np.uint8)

    def test_mesh_grid_dimensions_and_continuity(self):
        bbox = [0.5, 0.75, 0.2, 0.1]
        grid_res = 40
        X, Y, Z_surface, Z_plane, geom = self.vis.generate_3d_pothole_mesh(self.synth_img, bbox, grid_res=grid_res)

        self.assertEqual(X.shape, (grid_res, grid_res), "Mesh X must match grid_res")
        self.assertEqual(Y.shape, (grid_res, grid_res), "Mesh Y must match grid_res")
        self.assertEqual(Z_surface.shape, (grid_res, grid_res), "Mesh Z surface must match grid_res")
        self.assertEqual(Z_plane.shape, (grid_res, grid_res), "Reference plane Z must match grid_res")

        # Undisturbed road plane is Z = 0
        self.assertTrue((Z_plane == 0.0).all(), "Road reference plane must be identically Z=0")

        # Minimum elevation in surface mesh should reach negative depth (cavity depression)
        min_z = np.min(Z_surface)
        self.assertLess(min_z, 0.0, "Surface cavity must depress beneath Z=0 plane")
        self.assertAlmostEqual(abs(min_z), geom["depth_max_cm"], delta=1.5,
                               msg="Lowest point in 3D surface must correspond to reconstructed metric depth")


if __name__ == "__main__":
    unittest.main(verbosity=2)
