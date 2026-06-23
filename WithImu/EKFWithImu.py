import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator

radius = 20
omega = 0.5
includeGPS = True
GPSvar = 1

# --- EKF Initialization ---
# State: [x, y, velocity, heading]
xe_hat = np.array([[radius], [0.0], [5.0], [np.pi/2]])
PE = np.eye(4) * 0.1

dt = 0.05
IMU = IMUSimulator(dt)

# Measurement Matrix (We only measure position via GPS)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Noise Covariances
QE = np.diag([0.01, 0.01, 0.1, 0.05]) # Uncertainty in physics
RE = np.eye(2) * 0.01  # Measurement noise (GPS variance: 1.5^2)

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
ekf_path, = ax1.plot([], [], 'k--', label='EKF')
drone_dot, = ax1.plot([], [], 'go', label='Truth')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()

history_ekf_x, history_ekf_y = [], []
rese_history_x, rese_history_y, rese_t = [], [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 300) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals (Innovation)")
ax2.set_ylabel("Error (m)")
rese_x_line, = ax2.plot([], [], 'k-', label='EKFX-Residual', alpha=0.6)
#rese_y_line, = ax2.plot([], [], 'k-', label='EKFY-Residual', alpha=0.6)
ax2.legend()

for i in range(300):
    # Extract current state for readability
    _, _, v, theta = xe_hat.flatten()
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    
    # 2. EKF PREDICT: Move the state forward using trig
    xe_hat[0,0] += v * np.cos(theta) * dt + 1.0 / 2 * a * np.cos(theta) * dt**2
    xe_hat[1,0] += v * np.sin(theta) * dt + 1.0 / 2 * a * np.sin(theta) * dt**2
    xe_hat[2,0] += a * dt
    xe_hat[3,0] += omegahat * dt
    
    # 2. LINEARIZE (The Jacobian G)
    # This is the derivative of the physics above
    G = np.array([
        [1, 0, np.cos(theta)*dt, -(v*np.sin(theta)*dt + 0.5 * a * np.sin(theta) * dt**2)],
        [0, 1, np.sin(theta)*dt,  (v*np.cos(theta)*dt + 0.5 * a * np.cos(theta) * dt**2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Update Covariance using the Jacobian
    PE = G @ PE @ G.T + QE

    # 3. KF UPDATE (Every 10 frames when GPS "arrives")
    if i % 10 == 0 and includeGPS:
        z = np.array([[curr_x + np.random.normal(0, math.sqrt(GPSvar))], 
                      [curr_y + np.random.normal(0, math.sqrt(GPSvar))]])
       
        # RESIDUAL calculation (This is the "Surprise")
        residuale = z - H @ xe_hat

        # Gain and Update
       
        SE = H @ PE @ H.T + RE
        KE = PE @ H.T @ np.linalg.inv(SE)
        xe_hat = xe_hat + KE @ (z - H @ xe_hat)
        PE = (np.eye(4) - KE @ H) @ PE
        
        # Store residuals for plotting
        rese_t.append(i)
        rese_history_x.append(residuale[0,0])
        rese_history_y.append(residuale[1,0])

        gps_dot.set_data([z[0,0]], [z[1,0]])

    # 4. Visualization
    history_ekf_x.append(xe_hat[0,0])
    history_ekf_y.append(xe_hat[1,0])
    
    ekf_path.set_data(history_ekf_x, history_ekf_y)
    drone_dot.set_data([curr_x], [curr_y])

    if rese_t: # Only plot if we have data
        rese_x_line.set_data(rese_t, rese_history_x)
        #rese_y_line.set_data(rese_t, rese_history_y)
    
    plt.pause(0.03)

plt.ioff(); plt.show()
