import numpy as np

# --- Simulation Parameters ---
fs = 100.0          # Sampling rate (Hz)
dt = 1.0 / fs
duration = 2 * 3600 # 2 hours in seconds
N = int(duration * fs)

# --- Define True Noise Parameters (What we want to find later) ---
# True Angle Random Walk (White Noise Density)
true_sigma_white = 0.005  # rad/s / sqrt(Hz)
# True Rate Random Walk (Bias Drift Density)
true_sigma_walk  = 0.0002 # rad/s^2 / sqrt(Hz)

print(f"Generating {N} samples of synthetic IMU data...")

# 1. Initialize arrays
gyro_measurements = np.zeros(N)
current_bias = 0.01  # Start with an initial static bias offset

# 2. Generate the data
for i in range(N):
    # Update the bias via random walk: multiply by sqrt(dt)
    current_bias += true_sigma_walk * np.sqrt(dt) * np.random.normal()
    
    # Generate instantaneous white noise: divide by sqrt(dt)
    gyro_fuzz = (true_sigma_white / np.sqrt(dt)) * np.random.normal()
    
    # Combined measurement (True angular velocity is 0)
    gyro_measurements[i] = 0.0 + current_bias + gyro_fuzz

# Save to file
np.savetxt("synthetic_gyro_data.txt", gyro_measurements)
print("Data saved to 'synthetic_gyro_data.txt'")
