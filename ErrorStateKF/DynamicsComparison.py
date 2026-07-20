import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator
from datetime import datetime

radius = 20
omega = 0.5
includeGPS = True
GPSoutage = 30 #seconds

# --- EKF Initialization ---
xes_hat = np.array([radius, 0.0, 5.0, np.pi/2]) #the nominal states do not include bias estimates 
xes2_hat = np.array([radius, 0.0, 5.0, np.pi/2]) #the nominal states do not include bias estimates 
error_states = np.array([[0], [0], [0], [0], [0], [0]])
error_states2 = np.array([[0], [0], [0], [0], [0], [0]])
Perror = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5])
Perror2 = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5])

dt = 0.01
totalTime = 1 #minutes
IMU = IMUSimulator(dt)

# Measurement Matrix (We only measure position via GPS)
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])

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
Q = np.diag([
    0.0, 0.0, 
    (sigma_accel_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accel_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt
])
R = np.eye(2) * GPS_var  # Measurement noise

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
drone_dot, = ax1.plot([], [], 'go', label='Truth')
eskf_path, = ax1.plot([], [], 'b--', label='ESKF')
eskf2_path, = ax1.plot([], [], 'k--', label='ESKF2')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_eskf2_x, history_eskf2_y = [], []
reses2_history_x, reses2_history_y, rese_t = [], [], []
reses_history_x, reses_history_y= [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals (Innovation)")
ax2.set_ylabel("Error (m)")
ax2.set_xlabel("Seconds")
reses2_x_line, = ax2.plot([], [], 'k-', label='EKF2X-Residual', alpha=0.6)
reses_x_line, = ax2.plot([], [], 'b-', label='ESKFX-Residual', alpha=0.6)
ax2.legend()
now = datetime.now()

I6 = np.eye(6)
for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corrES = a - error_states[4] # raw_accel - b_a
    w_corrES = omegahat - error_states[5] # raw_gyro - b_w
    a_corrES2 = a - error_states2[4] # raw_accel - b_a
    w_corrES2 = omegahat - error_states2[5] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    xES, yES, vES, thetaES = xes_hat
    xES2, yES2, vES2, thetaES2 = xes2_hat
    
    xES     += vES * np.cos(thetaES) * dt
    yES     += vES * np.sin(thetaES) * dt
    vES     += a_corrES[0] * dt
    thetaES += w_corrES[0] * dt
    xes_hat = np.array([xES, yES, vES, thetaES])

    xES2     += vES2 * np.cos(thetaES2) * dt + 1.0 / 2 * a_corrES2[0] * np.cos(thetaES2) * dt**2
    yES2     += vES2 * np.sin(thetaES2) * dt + 1.0 / 2 * a_corrES2[0] * np.sin(thetaES2) * dt**2
    vES2     += a_corrES2[0] * dt
    thetaES2 += w_corrES2[0] * dt
    xes2_hat = np.array([xES2, yES2, vES2, thetaES2])
    
    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = I6.copy()
    F[0, 2] = np.cos(thetaES) * dt
    F[0, 3] = -vES * np.sin(thetaES) * dt
    F[1, 2] = np.sin(thetaES) * dt
    F[1, 3] = vES * np.cos(thetaES) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt

    F2 = I6.copy()
    F2[0, 2] = np.cos(thetaES2) * dt
    F2[0, 3] = -vES2 * np.sin(thetaES2) * dt - 0.5 * a_corrES2[0] * np.sin(thetaES2) * dt**2
    F2[1, 2] = np.sin(thetaES2) * dt 
    F2[1, 3] = vES2 * np.cos(thetaES2) * dt + 0.5 * a_corrES2[0] * np.cos(thetaES2) * dt**2
    F2[2, 4] = -dt
    F2[3, 5] = -dt
    # Update Covariance using the Jacobian
    Perror = F @ Perror @ F.T + Q
    Perror2 = F2 @ Perror2 @ F2.T + Q


    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and includeGPS and i < (GPSoutage/dt):
        z = np.array([[curr_x + np.random.normal(0, math.sqrt(GPS_var))], 
                      [curr_y + np.random.normal(0, math.sqrt(GPS_var))]])
       
        # RESIDUAL calculation (This is the "Surprise")
        residuales = z - np.array([[xes_hat[0]], [xes_hat[1]]])
        residuales2 = z - np.array([[xes2_hat[0]], [xes2_hat[1]]])

        # Gain and Update
        Serror = H @ Perror @ H.T + R
        Kerror = Perror @ H.T @ np.linalg.inv(Serror)
        error_states = Kerror @ residuales
        Perror = (I6 - Kerror @ H) @ Perror
        
        Serror2 = H @ Perror2 @ H.T + R
        Kerror2 = Perror2 @ H.T @ np.linalg.inv(Serror2)
        error_states2 = Kerror2 @ residuales2
        Perror2 = (I6 - Kerror2 @ H) @ Perror2

        # --- INJECTION STEP ---
        # Apply error estimations directly to nominal totals
        xes_hat[0] += error_states[0,0]  # Fix X
        xes_hat[ 1] += error_states[1,0]  # Fix Y
        xes_hat[ 2] += error_states[2,0]  # Fix Velocity
        xes_hat[ 3] += error_states[3,0]  # Fix Heading
        
        xes2_hat[0] += error_states2[0,0]  # Fix X
        xes2_hat[ 1] += error_states2[1,0]  # Fix Y
        xes2_hat[ 2] += error_states2[2,0]  # Fix Velocity
        xes2_hat[ 3] += error_states2[3,0]  # Fix Heading
        # Store residuals for plotting
        rese_t.append(t)
        reses_history_x.append(residuales[0])
        reses_history_y.append(residuales[1])
        reses2_history_x.append(residuales2[0])
        reses2_history_y.append(residuales2[1])

        gps_dot.set_data([z[0,0]], [z[1,0]])
        print(datetime.now()- now)
        now = datetime.now()

    # 4. Visualization
    history_eskf_x.append(xes_hat[0])
    history_eskf_y.append(xes_hat[1])
    history_eskf2_x.append(xes2_hat[0])
    history_eskf2_y.append(xes2_hat[1])
    
    drone_dot.set_data([curr_x], [curr_y])
    eskf_path.set_data(history_eskf_x, history_eskf_y)
    eskf2_path.set_data(history_eskf2_x, history_eskf2_y)

    if rese_t: # Only plot if we have data
        reses_x_line.set_data(rese_t, reses_history_x)
        reses2_x_line.set_data(rese_t, reses2_history_x)
    
    plt.pause(0.001)
plt.ioff(); plt.show()
