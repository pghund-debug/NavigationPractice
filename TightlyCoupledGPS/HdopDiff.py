import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5

# --- EKF Initialization ---
x_hat = np.array([radius, 0.0, 5.0, np.pi/2, 0.0, 0.0, 45.0])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
Perror = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1e-4])

x_hat2 = np.array([radius, 0.0, 5.0, np.pi/2, 0.0, 0.0, 45.0])  
error_states2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
Perror2 = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1e-4])

dt = 0.01
totalTime = 1 #minutes
IMU = IMUSimulator(dt)
sat_angles1 = [30, 75, 120, 160, 220]
sat_angles2 = [30, 32, 34, 36]
GPS = GPSR(dt * 100, sat_angles1) # second argument is satellite angles in degrees
GPS2 = GPSR(dt * 100, sat_angles2)

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

sigma_clk_walk = 0.1

# Noise Covariances
Q = np.diag([
    0.0, 0.0, 
    (sigma_accel_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accel_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt,
    (sigma_clk_walk**2) * dt
])
R = np.eye(len(sat_angles1), dtype = float)
R2 = np.eye(len(sat_angles2), dtype = float)
# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
drone_dot, = ax1.plot([], [], 'go', label='Truth')
sat_plot, = ax1.plot([], [], 'ro', markersize=8, label="Visible Sats")
sat_plot2, = ax1.plot([], [], 'co', markersize=8, label="Visible Sats2")
eskf_path, = ax1.plot([], [], 'b--', label='ESKF')
eskf2_path, = ax1.plot([], [], 'm--', label='ESKF2')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_eskf2_x, history_eskf2_y = [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals 1")
ax2.set_ylabel("Error (m)")
ax2.grid(True)

ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax3.set_title("GPS Residuals 2")
ax3.set_ylabel("Error (m)")
ax3.grid(True)

# link a specific Satellite ID (PRN) to its data history and its plot line.
history_time = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_res = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objects = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_time2 = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_res2 = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objects2 = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corr = a - error_states[4] # raw_accel - b_a
    w_corr = omegahat - error_states[5] # raw_gyro - b_w
    a_corr2 = a - error_states2[4] # raw_accel - b_a
    w_corr2 = omegahat - error_states2[5] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    x, y, v, theta, ba, bw, bclk = x_hat
    x2, y2, v2, theta2, ba2, bw2, bclk2 = x_hat2
    
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    v     += a_corr * dt
    theta += w_corr * dt
    x_hat = np.array([x, y, v, theta, ba, bw, bclk])
    
    x2     += v2 * np.cos(theta2) * dt
    y2     += v2 * np.sin(theta2) * dt
    v2     += a_corr2 * dt
    theta2 += w_corr2 * dt
    x_hat2 = np.array([x2, y2, v2, theta2, ba2, bw2, bclk2])

    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = np.eye(7)
    F[0, 2] = np.cos(theta) * dt
    F[0, 3] = -v * np.sin(theta) * dt
    F[1, 2] = np.sin(theta) * dt
    F[1, 3] = v * np.cos(theta) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt
    
    F2 = np.eye(7)
    F2[0, 2] = np.cos(theta2) * dt
    F2[0, 3] = -v2 * np.sin(theta2) * dt
    F2[1, 2] = np.sin(theta2) * dt
    F2[1, 3] = v2 * np.cos(theta2) * dt
    F2[2, 4] = -dt
    F2[3, 5] = -dt

    # Update Covariance using the Jacobian
    Perror = F @ Perror @ F.T + Q
    Perror2 = F2 @ Perror2 @ F2.T + Q

    active_prns1=[0,1,2,3,4] #includes 0 as a PRN
    active_prns2=[0,1,2,3] #includes 0 as a PRN

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        print("GPS update") 
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawPRs2, estimated_sat_pos2, true_clock_bias2 = GPS2.get_satellite_positions(curr_x, curr_y)
        residual = np.zeros(len(sat_angles1))
        H = np.zeros((len(sat_angles1), 7))
        residual2 = np.zeros(len(sat_angles2))
        H2 = np.zeros((len(sat_angles2), 7))
        sat_x = []
        sat_y = []
        sat_x2 = []
        sat_y2 = []
    
        #batch incorporation of measurements
        for j in range(len(estimated_sat_pos)):
            if j in active_prns1:
                dx = x_hat[0] - estimated_sat_pos[j][0]
                dy = x_hat[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                H[j][0] = ux
                H[j][1] = uy
                H[j][6] = 1
                
                if j not in history_time:
                    history_time[j] = []
                    history_res[j] = []
                    # Create a new line with markers to see individual updates clearly
                    new_line, = ax2.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                    line_objects[j] = new_line
                    ax2.legend(loc='upper right') # Update legend with new sat
                
                pr_hat = r_nom + x_hat[6]
                ux*=28
                uy*=28
                residual[j] = rawPRs[j] - pr_hat
                
                # Append the current time and residual to this specific satellite's history
                history_time[j].append(t)
                history_res[j].append(residual[j])
                
                # Update the specific line object memory pointers
                line_objects[j].set_data(history_time[j], history_res[j])
                
                sat_x.append(ux)
                sat_y.append(uy)
        
        for j in range(len(estimated_sat_pos2)):
            if j in active_prns2:
                dx2 = x_hat2[0] - estimated_sat_pos2[j][0]
                dy2 = x_hat2[1] - estimated_sat_pos2[j][1]
                r_nom2 = np.sqrt(dx2**2 + dy2**2)
                ux2 = dx2 / r_nom2
                uy2 = dy2 / r_nom2
                H2[j][0] = ux2
                H2[j][1] = uy2
                H2[j][6] = 1

                if j not in history_time2:
                    history_time2[j] = []
                    history_res2[j] = []
                    # Create a new line with markers to see individual updates clearly
                    new_line, = ax3.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                    line_objects2[j] = new_line
                    ax3.legend(loc='upper right') # Update legend with new sat
                
                pr_hat2 = r_nom2 + x_hat2[6]
                ux2*=28
                uy2*=28
                residual2[j] = rawPRs2[j] - pr_hat2
                
                # Append the current time and residual to this specific satellite's history
                history_time2[j].append(t)
                history_res2[j].append(residual2[j])
                
                # Update the specific line object memory pointers
                line_objects2[j].set_data(history_time2[j], history_res2[j])

                sat_x2.append(ux2)
                sat_y2.append(uy2)


        sat_plot.set_data(sat_x, sat_y)
        sat_plot2.set_data(sat_x2, sat_y2) 
        K = Perror @ H.T @ np.linalg.inv (H @ Perror @ H.T + R)
        error_states = K @ residual.T
        Perror = (np.eye(7) -  K @ H) @ Perror
        
        K2 = Perror2 @ H2.T @ np.linalg.inv (H2 @ Perror2 @ H2.T + R2)
        error_states2 = K2 @ residual2.T
        Perror2 = (np.eye(7) -  K2 @ H2) @ Perror2
        # --- INJECTION STEP ---
        # Apply error estimations directly to nominal totals
        x_hat[0] += error_states[0]  # Fix X
        x_hat[ 1] += error_states[1]  # Fix Y
        x_hat[ 2] += error_states[2]  # Fix Velocity
        x_hat[ 3] += error_states[3]  # Fix Heading
        x_hat[ 4] = error_states[4]
        x_hat[ 5] = error_states[5]  
        
        x_hat2[0] += error_states2[0]  # Fix X
        x_hat2[ 1] += error_states2[1]  # Fix Y
        x_hat2[ 2] += error_states2[2]  # Fix Velocity
        x_hat2[ 3] += error_states2[3]  # Fix Heading
        x_hat2[ 4] = error_states2[4]
        x_hat2[ 5] = error_states2[5]  
        x_hat2[ 6] += error_states2[6]
        x_hat2[ 6] += error_states2[6]
        

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])
    
    history_eskf2_x.append(x_hat2[0])
    history_eskf2_y.append(x_hat2[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        eskf_path.set_data(history_eskf_x, history_eskf_y)
        eskf2_path.set_data(history_eskf2_x, history_eskf2_y)
        

    plt.pause(0.0001)
plt.ioff(); plt.show()
