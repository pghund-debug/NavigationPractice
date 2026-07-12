import numpy as np

class IMUSimulator:
    def __init__(self, dt):
        self.dt = dt
        
        # random walk is an integrated white noise
        # --- Gyroscope Parameters (Typical low-cost MEMS) ---
        self.gyro_white_noise_sigma = 0.005  # rad/s / sqrt(Hz)
        self.gyro_bias_walk_sigma  = 0.0008 # rad/s^2 / sqrt(Hz)
        self.gyro_bias = 0.01                # Initial turn rate bias (rad/s)
        
        # --- Accelerometer Parameters ---
        self.accel_white_noise_sigma = 0.05  # m/s^2 / sqrt(Hz)
        self.accel_bias_walk_sigma   = 0.008 # m/s^3 / sqrt(Hz)
        self.accel_bias = 0.1                # Initial acceleration bias (m/s^2)
        
    def generate_measurements(self, true_a_body, true_omega):
        """
        Takes ideal truth values and corrupts them with noise and drifting bias.
        """
        # 1. Update Drifting Biases (Random Walk Model)
        # bias_k = bias_{k-1} + white_noise * sqrt(dt)
        # Why multiply? Because a random walk drifts further
        # the more time that passes during a step.
        self.gyro_bias  += np.random.normal(0, self.gyro_bias_walk_sigma) * np.sqrt(self.dt)
        self.accel_bias += np.random.normal(0, self.accel_bias_walk_sigma) * np.sqrt(self.dt)
        
        # 2. Add Instantaneous White Noise & Bias to Truth
        # Why divide? Because smaller time steps sample the noise faster,
        # which requires higher variance per step to keep the physics consistent.
        # noise = sigma * normal_dist / sqrt(dt)
        gyro_noise  = (self.gyro_white_noise_sigma / np.sqrt(self.dt)) * np.random.normal()
        accel_noise = (self.accel_white_noise_sigma / np.sqrt(self.dt)) * np.random.normal()
        
        # 3. Final Measured Signals
        measured_omega  = true_omega + self.gyro_bias + gyro_noise
        measured_a_body = true_a_body + self.accel_bias + accel_noise
        
        return measured_a_body, measured_omega, self.gyro_bias, self.accel_bias
