import allantools
import matplotlib.pyplot as plt
import numpy as np

# Load the generated synthetic data
data = np.loadtxt("synthetic_gyro_data.txt")
fs = 100.0  # Sampling frequency

print("Calculating Allan Deviation (this may take a moment)...")
# Calculate Allan Deviation
# 'local_rate' tells the library we are passing raw gyro rates (rad/s), not integrated angles
taus, adev, errors, ns = allantools.oadev(data, rate=fs, data_type="freq")

# Plotting the Results
plt.figure(figsize=(10, 6))
plt.loglog(taus, adev, label='Calculated Allan Deviation', color='blue', linewidth=2)

# Add helper lines to identify the slopes visually
# 1. White Noise dominates the left side with a -0.5 slope (1/sqrt(tau))
plt.loglog([0.01, 10], [0.05, 0.0005], '--', color='gray', label='White Noise Slope (-0.5)')

# 2. Random Walk dominates the right side with a +0.5 slope (sqrt(tau))
plt.loglog([10, 1000], [0.0001, 0.001], ':', color='red', label='Bias Walk Slope (+0.5)')

plt.title('Allan Deviation Analysis of Synthetic Gyroscope Data')
plt.xlabel('Cluster Time $\\tau$ (seconds)')
plt.ylabel('Allan Deviation $\\sigma(\\tau)$ (rad/s)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()
