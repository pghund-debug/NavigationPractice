import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator

radius = 20
omega = 0.5
includeGPS = True
plotBiases = False

# --- EKF Initialization ---
# State: [x, y, velocity, heading]
xe_hat = np.array([[radius], [0.0], [5.0], [np.pi/2]])
xe_hat2 = np.array([[radius], [0.0], [5.0], [np.pi/2], [0.1], [0.1]])
PE = np.eye(4) * 0.1
PE2 = np.eye(6) * 0.1

dt = 0.05
totalTime = 5 #minutes
IMU = IMUSimulator(dt)

# Measurement Matrix (We only measure position via GPS)
H2 = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

sigma_accel_white = 0.04
sigma_gyro_white = 0.006

sigma_accel_walk = 0.001
sigma_gyro_walk = 0.0001

# Standard White Noise Variances (from the tau=1 intercept)
var_v     = (sigma_accel_white ** 2) * dt
var_theta = (sigma_gyro_white ** 2) * dt

# Bias Drift Variances (from the sloped right side of the Allan plot)
var_ba_walk = (sigma_accel_walk ** 2) * dt
var_bw_walk = (sigma_gyro_walk ** 2) * dt

GPS_var = 2.25

# Noise Covariances
QE = np.diag([0.0, 0.0, var_v, var_theta]) # Uncertainty in physics
QE2 = np.diag([0.0, 0.0, var_v, var_theta, var_ba_walk, var_bw_walk]) # Uncertainty in physics
RE = np.eye(2) * GPS_var  # Measurement noise

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
#ekf_path, = ax1.plot([], [], 'k--', label='EKF')
#ekf_path2, = ax1.plot([], [], 'b--', label='EKFB')
drone_dot, = ax1.plot([], [], 'go', label='Truth')
EKF_dot, = ax1.plot([], [], 'ko', label='EKF')
EKF2_dot, = ax1.plot([], [], 'bo', label='EKFB')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()

history_ekf_x, history_ekf_y = [], []
history_ekf2_x, history_ekf2_y = [], []
rese_history_x, rese_history_y, rese_t = [], [], []
rese2_history_x, rese2_history_y= [], []
history_gyrob, history_accb = [], []
history_truegyrob, history_trueaccb = [], []

if plotBiases:
    ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
    ax2.set_xlabel("Seconds")
    ax2.set_ylim(-0.05, 0.16)
    ax2.set_title("bias estimate")
    acc_line, = ax2.plot([], [], 'g-', label='acc bias estimate', alpha=0.6)
    gyro_line, = ax2.plot([], [], 'b-', label='gyro bias estimate', alpha=0.6)
    trueacc_line, = ax2.plot([], [], 'g:', label='true acc bias', alpha=0.6)
    truegyro_line, = ax2.plot([], [], 'b:', label='true gyro bias', alpha=0.6)
    ax2.legend()

else:
    # Bottom Plot: Residuals (z - Hx)
    ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
    ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
    ax2.set_title("GPS Residuals (Innovation)")
    ax2.set_ylabel("Error (m)")
    rese_x_line, = ax2.plot([], [], 'k-', label='EKFX-Residual', alpha=0.6)
    #rese_y_line, = ax2.plot([], [], 'k-', label='EKFY-Residual', alpha=0.6)
    rese2_x_line, = ax2.plot([], [], 'b-', label='EKFBX-Residual', alpha=0.6)
    #rese2_y_line, = ax2.plot([], [], 'b-', label='EKFBY-Residual', alpha=0.6)
    ax2.legend()

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    _, _, v, theta = xe_hat.flatten()
    _, _, v2, theta2, _,_ = xe_hat2.flatten()
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corr = a - xe_hat2[4,0] # raw_accel - b_a
    w_corr = omegahat - xe_hat2[5,0] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    xe_hat[0,0] += v * np.cos(theta) * dt + 1.0 / 2 * a * np.cos(theta) * dt**2
    xe_hat[1,0] += v * np.sin(theta) * dt + 1.0 / 2 * a * np.sin(theta) * dt**2
    xe_hat[2,0] += a * dt
    xe_hat[3,0] += omegahat * dt
    
    xe_hat2[0,0] += v2 * np.cos(theta2) * dt + 1.0 / 2 * a_corr * np.cos(theta2) * dt**2
    xe_hat2[1,0] += v2 * np.sin(theta2) * dt + 1.0 / 2 * a_corr * np.sin(theta2) * dt**2
    xe_hat2[2,0] += a_corr * dt
    xe_hat2[3,0] += w_corr * dt
    
    # 2. LINEARIZE (The Jacobian G)
    # This is the derivative of the physics above
    G = np.array([
        [1, 0, np.cos(theta)*dt, -(v*np.sin(theta)*dt + 0.5 * a * np.sin(theta) * dt**2)],
        [0, 1, np.sin(theta)*dt,  (v*np.cos(theta)*dt + 0.5 * a * np.cos(theta) * dt**2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    G2 = np.array([
        [1, 0, np.cos(theta2)*dt, -(v2*np.sin(theta2)*dt + 0.5 * a * np.sin(theta2) * dt**2), 0, 0],
        [0, 1, np.sin(theta2)*dt,  (v2*np.cos(theta2)*dt + 0.5 * a * np.cos(theta2) * dt**2), 0, 0],
        [0, 0, 1, 0, -dt, 0],
        [0, 0, 0, 1, 0, -dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    # Update Covariance using the Jacobian
    PE = G @ PE @ G.T + QE
    PE2 = G2 @ PE2 @ G2.T + QE2

    # 3. KF UPDATE (Every 10 frames when GPS "arrives")
    if i % 10 == 0 and includeGPS:
        z = np.array([[curr_x + np.random.normal(0, math.sqrt(GPS_var))], 
                      [curr_y + np.random.normal(0, math.sqrt(GPS_var))]])
       
        # RESIDUAL calculation (This is the "Surprise")
        residuale = z - H @ xe_hat
        residuale2 = z - H2 @ xe_hat2

        # Gain and Update
        SE = H @ PE @ H.T + RE
        KE = PE @ H.T @ np.linalg.inv(SE)
        xe_hat = xe_hat + KE @ (z - H @ xe_hat)
        PE = (np.eye(4) - KE @ H) @ PE
        
        SE2 = H2 @ PE2 @ H2.T + RE
        KE2 = PE2 @ H2.T @ np.linalg.inv(SE2)
        xe_hat2 = xe_hat2 + KE2 @ (z - H2 @ xe_hat2)
        PE2 = (np.eye(6) - KE2 @ H2) @ PE2
        
        # Store residuals for plotting
        rese_t.append(i * 0.05)
        rese_history_x.append(residuale[0,0])
        rese_history_y.append(residuale[1,0])
        rese2_history_x.append(residuale2[0,0])
        rese2_history_y.append(residuale2[1,0])

        history_accb.append(xe_hat2[4,0])
        history_gyrob.append(xe_hat2[5,0])
        history_trueaccb.append(trueAccelBias)
        history_truegyrob.append(trueGyroBias)
        gps_dot.set_data([z[0,0]], [z[1,0]])

    # 4. Visualization
    history_ekf_x.append(xe_hat[0,0])
    history_ekf_y.append(xe_hat[1,0])
    history_ekf2_x.append(xe_hat2[0,0])
    history_ekf2_y.append(xe_hat2[1,0])
    
    #ekf_path.set_data(history_ekf_x, history_ekf_y)
    #ekf_path2.set_data(history_ekf2_x, history_ekf2_y)
    drone_dot.set_data([curr_x], [curr_y])
    EKF_dot.set_data([xe_hat[0,0]], [xe_hat[1,0]])
    EKF2_dot.set_data([xe_hat2[0,0]], [xe_hat2[1,0]])

    if plotBiases:
        if rese_t:
            acc_line.set_data(rese_t, history_accb)
            gyro_line.set_data(rese_t, history_gyrob)
            trueacc_line.set_data(rese_t, history_trueaccb)
            truegyro_line.set_data(rese_t, history_truegyrob)

    else:
        if rese_t: # Only plot if we have data
            rese_x_line.set_data(rese_t, rese_history_x)
            rese2_x_line.set_data(rese_t, rese2_history_x)
            #rese_y_line.set_data(rese_t, rese_history_y)
            #rese_y_line2.set_data(rese_t, rese2_history_y)
    
  #  plt.pause(0.01)
plt.ioff(); plt.show()
