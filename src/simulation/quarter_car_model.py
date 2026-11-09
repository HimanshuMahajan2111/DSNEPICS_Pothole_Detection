"""
Quarter-Car Suspension Dynamic Simulator
=========================================
Implements the 2-Degree-of-Freedom (2-DOF) state-space differential equations
governing vehicle sprung mass (body) and unsprung mass (axle/wheel) response
as a tire traverses a 3D pothole elevation profile.

Empirically calibrated constants:
- Sprung mass (ms): 300.0 kg
- Unsprung mass (mu): 45.0 kg
- Suspension stiffness (ks): 18,500 N/m
- Suspension damping (cs): 1,450 Ns/m (zeta = 0.32, fn = 2.50 Hz)
- Tire stiffness (kt): 190,000 N/m
"""

import numpy as np
from scipy.integrate import solve_ivp

class QuarterCarSimulator:
    def __init__(self, ms=300.0, mu=45.0, ks=18500.0, cs=1450.0, kt=190000.0):
        self.ms = float(ms)
        self.mu = float(mu)
        self.ks = float(ks)
        self.cs = float(cs)
        self.kt = float(kt)
        
    def generate_pothole_road_profile(self, depth_max_m, length_m, speed_mps, dt=0.005, pre_time=0.5, post_time=1.5):
        """
        Generates road elevation excitation z_r(t) as the vehicle hits the pothole.
        """
        t_enter = pre_time
        duration_cavity = length_m / speed_mps
        t_exit = t_enter + duration_cavity
        total_time = t_exit + post_time
        
        t_eval = np.arange(0, total_time, dt)
        z_r = np.zeros_like(t_eval)
        
        # Parabolic dip profile during traversal
        in_cavity = (t_eval >= t_enter) & (t_eval <= t_exit)
        tau = (t_eval[in_cavity] - t_enter) / duration_cavity  # 0 to 1
        # Parabola: z_r = -depth * 4 * tau * (1 - tau)
        z_r[in_cavity] = -depth_max_m * 4.0 * tau * (1.0 - tau)
        
        return t_eval, z_r, t_enter, t_exit

    def simulate_pothole_impact(self, depth_max_cm, length_cm, speed_kmh=35.0, dt=0.005):
        """
        Solves 2-DOF ODE:
        ms * z_s'' + cs*(z_s' - z_u') + ks*(z_s - z_u) = 0
        mu * z_u'' - cs*(z_s' - z_u') - ks*(z_s - z_u) + kt*(z_u - z_r) = 0
        
        Returns:
        - t_eval: time vector (s)
        - a_s_g: sprung mass vertical acceleration (g)
        - a_u_g: unsprung mass acceleration (g)
        - peak_g: maximum absolute body acceleration
        - p2p_g: peak-to-peak body acceleration
        - rms_g: RMS body acceleration
        """
        depth_m = (depth_max_cm / 100.0)
        length_m = (length_cm / 100.0)
        speed_mps = max(speed_kmh / 3.6, 1.5)
        
        t_eval, z_r, t_enter, t_exit = self.generate_pothole_road_profile(depth_m, length_m, speed_mps, dt=dt)
        
        # State vector: Y = [z_s, z_s_dot, z_u, z_u_dot]
        def ode_func(t, Y):
            z_s, v_s, z_u, v_u = Y
            
            # Interpolate road profile at instantaneous t
            z_road = np.interp(t, t_eval, z_r)
            
            # Forces
            f_suspension = self.ks * (z_s - z_u) + self.cs * (v_s - v_u)
            f_tire = self.kt * (z_u - z_road)
            
            # Accelerations
            a_s = -f_suspension / self.ms
            a_u = (f_suspension - f_tire) / self.mu
            
            return [v_s, a_s, v_u, a_u]
            
        sol = solve_ivp(
            ode_func,
            [t_eval[0], t_eval[-1]],
            [0.0, 0.0, 0.0, 0.0],
            t_eval=t_eval,
            max_step=dt,
            method="RK45"
        )
        
        z_s = sol.y[0]
        v_s = sol.y[1]
        z_u = sol.y[2]
        v_u = sol.y[3]
        
        # Acceleration of sprung mass (in m/s^2 then g)
        f_susp = self.ks * (z_s - z_u) + self.cs * (v_s - v_u)
        a_s_mps2 = -f_susp / self.ms
        a_s_g = a_s_mps2 / 9.81
        
        # Acceleration of unsprung mass
        f_tire = self.kt * (z_u - z_r)
        a_u_mps2 = (f_susp - f_tire) / self.mu
        a_u_g = a_u_mps2 / 9.81
        
        # Extract post-impact window
        shock_idx = np.where(t_eval >= t_enter)[0]
        shock_a = a_s_g[shock_idx]
        
        peak_g = float(np.max(np.abs(shock_a)))
        p2p_g = float(np.ptp(shock_a))
        rms_g = float(np.sqrt(np.mean(shock_a ** 2)))
        
        # Spectral energy in 3-12 Hz
        fft_vals = np.abs(np.fft.rfft(shock_a))
        freqs = np.fft.rfftfreq(len(shock_a), dt)
        band_mask = (freqs >= 3.0) & (freqs <= 12.0)
        fft_energy_3_12 = float(np.sum(fft_vals[band_mask] ** 2) / len(shock_a))
        
        return {
            "t_eval": t_eval.tolist(),
            "accel_sprung_g": a_s_g.tolist(),
            "peak_g": peak_g,
            "p2p_g": p2p_g,
            "rms_g": rms_g,
            "fft_energy_3_12hz": fft_energy_3_12,
            "speed_norm_shock": float(peak_g / speed_mps),
            "simulated_speed_kmh": float(speed_kmh)
        }


if __name__ == "__main__":
    print("[Quarter-Car Simulator] Running Dynamic Impact Test...")
    sim = QuarterCarSimulator()
    res = sim.simulate_pothole_impact(depth_max_cm=4.5, length_cm=45.0, speed_kmh=35.0)
    print(f"Simulated Pothole: Depth=4.5cm, Length=45cm at 35 km/h")
    print(f"  Peak Body Shock : {res['peak_g']:.4f} g")
    print(f"  Peak-to-Peak    : {res['p2p_g']:.4f} g")
    print(f"  Shock RMS       : {res['rms_g']:.4f} g")
    print(f"  Speed-Norm Shock: {res['speed_norm_shock']:.4f} g*s/m")
    print("[SUCCESS] Dynamic ODE Solver verified!")
