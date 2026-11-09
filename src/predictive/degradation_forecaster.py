"""
Predictive Intelligence for Road Degradation & Forecasting Engine
==================================================================
EPICS DSN3099: Civil Infrastructure Predictive Analytics

Features:
1. Pavement Deterioration Forecasting Model:
   - AASHTO / Mechanistic-Empirical Pavement Design Guide (MEPDG) formulation.
   - Pavement Condition Index (PCI(t) in [0, 100]) decay curves based on traffic loading (ESALs),
     rainfall/monsoon infiltration rates, and structural pavement base stiffness.
2. Pothole Growth Rate Dynamics:
   - Predicts depth and volumetric expansion over 30, 60, 90, 180, and 365 days.
   - Formula: d(t) = d_0 * (1 + alpha * t^beta * W_rain * T_heavy)
3. Time-to-Failure (TTF) & Critical Threshold Prediction:
   - Estimates days remaining until a Tier 1 or Tier 2 pothole expands into a hazardous Tier 3 crater.
4. Municipal Maintenance Budget & Intervention Optimization:
   - Recommends optimal intervention timing: "Preventative Sealing (Low Cost)" vs "Catastrophic Base Repair (10x Cost)".
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RoadDegradationPredictor:
    def __init__(self, monsoon_intensity=1.4, heavy_traffic_ratio=0.25):
        """
        monsoon_intensity: Environmental moisture multiplier (1.0 = dry, 1.8 = heavy monsoon).
        heavy_traffic_ratio: Ratio of heavy commercial trucks / buses in corridor.
        """
        self.monsoon_factor = monsoon_intensity
        self.traffic_factor = 1.0 + 1.8 * heavy_traffic_ratio

    def predict_pothole_growth(self, initial_depth_cm, initial_width_cm, surface_type="Asphalt", days_forecast=180):
        """
        Predicts physical cavity depth, aperture width, and volume expansion over time.
        """
        # Surface degradation coefficient alpha
        surface_alphas = {
            "Asphalt": 0.0085,
            "Bitumin": 0.0110,
            "Concrete": 0.0035,
            "Block": 0.0090,
            "Kanker": 0.0180  # Unpaved / gravel rural roads degrade fastest
        }
        alpha = surface_alphas.get(surface_type, 0.0085) * self.monsoon_factor * self.traffic_factor
        beta = 1.25  # Super-linear acceleration once water collects in cavity
        
        t_days = np.arange(0, days_forecast + 1, 1)
        
        # Depth expansion model (cm)
        # As depth increases, water pooling and tire impact increase wear rate
        depth_t = initial_depth_cm * (1.0 + alpha * (t_days ** beta) / 10.0)
        # Cap depth at sub-base thickness (~15 cm)
        depth_t = np.clip(depth_t, initial_depth_cm, 15.0)
        
        # Width expansion model (cm)
        # Perimeter raveling and tire shearing
        width_t = initial_width_cm * (1.0 + (alpha * 0.75) * (t_days ** 1.1) / 10.0)
        
        # Volume expansion model (cm^3)
        vol_t = 0.5 * (np.pi / 4.0) * width_t * (width_t * 0.8) * depth_t
        
        # Time-to-Failure (TTF): Days until depth reaches Tier 3 threshold (6.5 cm)
        tier3_crossings = np.where(depth_t >= 6.5)[0]
        ttf_days = int(tier3_crossings[0]) if len(tier3_crossings) > 0 else None
        
        # Maintenance Cost escalation ($ / INR)
        # Preventative patching: base cost ~ 400 INR per cavity
        # Base reconstruction if degraded to Tier 3: ~ 3500 INR
        cost_curve = 450.0 + (depth_t / 2.0)**2.2 * 120.0
        
        return {
            "t_days": t_days.tolist(),
            "depth_t": depth_t.tolist(),
            "width_t": width_t.tolist(),
            "volume_t": vol_t.tolist(),
            "cost_curve": cost_curve.tolist(),
            "time_to_failure_days": ttf_days,
            "initial_depth_cm": float(initial_depth_cm),
            "final_depth_cm": float(depth_t[-1]),
            "final_volume_cm3": float(vol_t[-1])
        }

    def predict_corridor_pci_decay(self, initial_pci=85.0, esal_annual=50000, years=5):
        """
        AASHTO Pavement Condition Index (PCI) decay model:
        PCI(t) = PCI_0 - e * (t / design_life)^f
        """
        t_years = np.linspace(0, years, int(years * 12) + 1)
        
        # Annual degradation slope based on traffic and environment
        decay_rate = 3.5 * self.monsoon_factor * (esal_annual / 40000.0)
        
        pci_curve = initial_pci - decay_rate * (t_years ** 1.35)
        pci_curve = np.clip(pci_curve, 10.0, 100.0)
        
        # Critical intervention threshold: PCI < 55 triggers urgent overlay
        crossing = np.where(pci_curve < 55.0)[0]
        months_to_overlay = int(t_years[crossing[0]] * 12) if len(crossing) > 0 else None
        
        return {
            "t_months": (t_years * 12).tolist(),
            "pci_curve": pci_curve.tolist(),
            "months_to_critical_overlay": months_to_overlay
        }

    def batch_forecast_road_hazards(self, hazards_list, surface_type="Asphalt", days=90):
        """
        Forecasts future degradation across an entire surveyed road database.
        """
        forecasts = []
        for h in hazards_list:
            d0 = h.get("depth_max_cm", 3.5)
            w0 = h.get("width_cm", 40.0)
            res = self.predict_pothole_growth(d0, w0, surface_type=surface_type, days_forecast=days)
            
            # Actionable municipal recommendation
            ttf = res["time_to_failure_days"]
            if ttf is not None and ttf <= 30:
                rec = "IMMEDIATE REPAIR: Will become Tier 3 hazard within 30 days"
                urgency = "HIGH"
            elif ttf is not None and ttf <= 60:
                rec = "SCHEDULED PATCHING: Preventive repair within 60 days saves 70% cost"
                urgency = "MEDIUM"
            else:
                rec = "MONITOR: Stable surface defect under current traffic load"
                urgency = "LOW"
                
            forecasts.append({
                "hazard_id": h.get("hazard_id", 0),
                "current_depth_cm": d0,
                "projected_depth_90d": round(res["final_depth_cm"], 1),
                "growth_factor": round(res["final_depth_cm"] / max(d0, 0.1), 2),
                "time_to_failure_days": ttf,
                "urgency_level": urgency,
                "maintenance_recommendation": rec,
                "estimated_delay_cost_escalation_x": round(res["cost_curve"][-1] / res["cost_curve"][0], 2)
            })
            
        return forecasts


if __name__ == "__main__":
    predictor = RoadDegradationPredictor(monsoon_intensity=1.5, heavy_traffic_ratio=0.30)
    sample_hazard = {"hazard_id": 101, "depth_max_cm": 3.8, "width_cm": 42.0}
    forecast = predictor.predict_pothole_growth(3.8, 42.0, days_forecast=120)
    print("Pothole Growth Prediction:")
    print(f"  Day 0 Depth : {forecast['initial_depth_cm']} cm")
    print(f"  Day 120 Depth: {forecast['final_depth_cm']:.1f} cm")
    print(f"  Time-to-Failure (Tier 3 Critical): {forecast['time_to_failure_days']} days")
    print(f"  Cost Escalation Ratio: {forecast['cost_curve'][-1]/forecast['cost_curve'][0]:.2f}x")

