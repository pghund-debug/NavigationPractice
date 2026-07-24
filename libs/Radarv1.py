import numpy as np

class RadarSimulator:
    def __init__(self, radar_x, radar_y, dt):
        """
        Initialize the ground-based tracking radar.
        radar_x, radar_y: The global coordinates of the radar station.
        dt: The time step of the radar measurement (e.g., 0.1s for 10Hz).
        """
        self.radar_x = radar_x
        self.radar_y = radar_y
        self.dt = dt
        
        # --- Radar Hardware Noise Specs ---
        self.sigma_r = 2.0                 # Range noise (meters)
        self.sigma_phi = np.radians(0.5)   # Azimuth noise (0.5 degrees in radians)
        self.sigma_rr = 0.1                # Range Rate / Doppler noise (m/s)

    def get_measurements(self, true_x, true_y, true_vx, true_vy):
        """
        Generates noisy radar observations based on the drone's true state.
        """
        # 1. Calculate True Geometry
        dx = true_x - self.radar_x
        dy = true_y - self.radar_y
        
        r_true = np.sqrt(dx**2 + dy**2)
        
        # Protect against division by zero if the drone crashes into the radar
        if r_true < 1e-6:
            r_true = 1e-6
            
        ux = dx / r_true
        uy = dy / r_true
        
        phi_true = np.arctan2(dy, dx)
        
        # True relative velocity (Doppler shift)
        rr_true = (ux * true_vx) + (uy * true_vy)
        
        # 2. Corrupt with Gaussian Sensor Noise
        meas_r = r_true + np.random.normal(0, self.sigma_r)
        meas_phi = phi_true + np.random.normal(0, self.sigma_phi)
        meas_rr = rr_true + np.random.normal(0, self.sigma_rr)
        
        # 3. Angle Wrapping
        # Ensures the noisy angle doesn't accidentally jump from 179 deg to 181 deg,
        # keeping it strictly bounded between -pi and +pi.
        meas_phi = (meas_phi + np.pi) % (2 * np.pi) - np.pi
        
        return meas_r, meas_phi, meas_rr
