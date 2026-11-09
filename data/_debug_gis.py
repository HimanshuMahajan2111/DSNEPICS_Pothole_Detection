import sys; sys.path.insert(0, r'd:\DSN3099-EPICS\Implementation')
# Check full gis mapper - what does build_interactive_gis_html take?
import inspect
from src.gis.road_mapper import GISRoadConditionMapper
mapper = GISRoadConditionMapper()
print('build_interactive_gis_html sig:', inspect.signature(mapper.build_interactive_gis_html))

# Also check the forecaster TTF for shallow potholes
from src.predictive.degradation_forecaster import RoadDegradationPredictor
fc = RoadDegradationPredictor()
r = fc.predict_pothole_growth(initial_depth_cm=2.0, initial_width_cm=25.0)
print('TTF for 2cm:', r['time_to_failure_days'])
r2 = fc.predict_pothole_growth(initial_depth_cm=5.0, initial_width_cm=50.0)
print('TTF for 5cm:', r2['time_to_failure_days'])
r3 = fc.predict_pothole_growth(initial_depth_cm=10.0, initial_width_cm=80.0)
print('TTF for 10cm:', r3['time_to_failure_days'])
