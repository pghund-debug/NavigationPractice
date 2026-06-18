import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator

radius = 20
omega = 0.5
includeGPS = True
plotBiases = False

# --- EKF Initialization ---
xe_hat = np.array([[radius], [0.0], [5.0], [np.pi/2], [0.1], [0.1]])
xes_hat = np.array([radius, 0.0, 5.0, np.pi/2]) #the nominal states do not include bias estimates 
error_states = np.array([[0], [0], [0], [0], [0], [0]])
P = np.eye(6) * 0.1
Perror = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5])

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
EKF_dot, = ax1.plot([], [], 'ko', label='EKF')
ESKF_dot, = ax1.plot([], [], 'bo', label='ESKF')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()

history_ekf_x, history_ekf_y = [], []
history_eskf_x, history_eskf_y = [], []
rese_history_x, rese_history_y, rese_t = [], [], []
reses_history_x, reses_history_y= [], []
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
    reses_x_line, = ax2.plot([], [], 'b-', label='ESKFX-Residual', alpha=0.6)
    #reses_y_line, = ax2.plot([], [], 'b-', label='ESKFY-Residual', alpha=0.6)
    ax2.legend()

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    _, _, v, theta, _,_ = xe_hat.flatten()
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corr = a - xe_hat[4,0] # raw_accel - b_a
    w_corr = omegahat - xe_hat[5,0] # raw_gyro - b_w
    a_corrES = a - error_states[4] # raw_accel - b_a
    w_corrES = omegahat - error_states[5] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    xe_hat[0,0] += v * np.cos(theta) * dt 
    xe_hat[1,0] += v * np.sin(theta) * dt 
    xe_hat[2,0] += a_corr * dt
    xe_hat[3,0] += w_corr * dt
    
    xES, yES, vES, thetaES = xes_hat
    
    xES     += vES * np.cos(thetaES) * dt
    yES     += vES * np.sin(thetaES) * dt
    vES     += a_corrES[0] * dt
    thetaES += w_corrES[0] * dt
    xes_hat = np.array([xES, yES, vES, thetaES])

    # 2. LINEARIZE (The Jacobian G)
    # This is the derivative of the physics above
    G = np.array([
        [1, 0, np.cos(theta)*dt, -(v*np.sin(theta)*dt ), 0, 0],
        [0, 1, np.sin(theta)*dt,  (v*np.cos(theta)*dt ), 0, 0],
        [0, 0, 1, 0, -dt, 0],
        [0, 0, 0, 1, 0, -dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    F = np.eye(6)
    F[0, 2] = np.cos(thetaES) * dt
    F[0, 3] = -vES * np.sin(thetaES) * dt
    F[1, 2] = np.sin(thetaES) * dt
    F[1, 3] = vES * np.cos(thetaES) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt

    # Update Covariance using the Jacobian
    P = G @ P @ G.T + Q
    Perror = F @ Perror @ F.T + Q

    # 3. KF UPDATE (Every 10 frames when GPS "arrives")
    if i % 10 == 0 and includeGPS:
        z = np.array([[curr_x + np.random.normal(0, math.sqrt(GPS_var))], 
                      [curr_y + np.random.normal(0, math.sqrt(GPS_var))]])
       
        # RESIDUAL calculation (This is the "Surprise")
        residuale = z - H @ xe_hat
        residuales = z - np.array([[xes_hat[0]], [xes_hat[1]]])

        # Gain and Update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        xe_hat = xe_hat + K @ (z - H @ xe_hat)
        P = (np.eye(6) - K @ H) @ P
        
        Serror = H @ Perror @ H.T + R
        Kerror = Perror @ H.T @ np.linalg.inv(Serror)
        error_states = Kerror @ residuales
        Perror = (np.eye(6) - Kerror @ H) @ Perror
        # --- INJECTION STEP ---
        # Apply error estimations directly to nominal totals
        xes_hat[0] += error_states[0,0]  # Fix X
        xes_hat[ 1] += error_states[1,0]  # Fix Y
        xes_hat[ 2] += error_states[2,0]  # Fix Velocity
        xes_hat[ 3] += error_states[3,0]  # Fix Heading
        # Store residuals for plotting
        rese_t.append(i * 0.05)
        rese_history_x.append(residuale[0,0])
        rese_history_y.append(residuale[1,0])
        reses_history_x.append(residuales[0])
        reses_history_y.append(residuales[1])

        history_accb.append(xe_hat[4,0])
        history_gyrob.append(xe_hat[5,0])
        history_trueaccb.append(trueAccelBias)
        history_truegyrob.append(trueGyroBias)
        gps_dot.set_data([z[0,0]], [z[1,0]])

    # 4. Visualization
    history_ekf_x.append(xe_hat[0,0])
    history_ekf_y.append(xe_hat[1,0])
    history_eskf_x.append(xes_hat[0])
    history_eskf_y.append(xes_hat[0])
    
    drone_dot.set_data([curr_x], [curr_y])
    EKF_dot.set_data([xe_hat[0,0]], [xe_hat[1,0]])
    ESKF_dot.set_data([xes_hat[0]], [xes_hat[1]])

    if plotBiases:
        if rese_t:
            acc_line.set_data(rese_t, history_accb)
            gyro_line.set_data(rese_t, history_gyrob)
            trueacc_line.set_data(rese_t, history_trueaccb)
            truegyro_line.set_data(rese_t, history_truegyrob)

    else:
        if rese_t: # Only plot if we have data
            rese_x_line.set_data(rese_t, rese_history_x)
            reses_x_line.set_data(rese_t, reses_history_x)
            #rese_y_line.set_data(rese_t, rese_history_y)
            #reses_y_line.set_data(rese_t, reses_history_y)
    
    plt.pause(0.01)
plt.ioff(); plt.show()
