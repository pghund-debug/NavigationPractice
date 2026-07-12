import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv2 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5

#state: x, y, vx, vy, theta, b_ax, b_ay, b_w, b_clk, b_drift
x_hat = np.array([radius, 0.0, -radius * omega * np.sin(0), radius * omega * np.cos(0), np.pi/2, 0.0, 0.0, 0.0, 45.0, 0.14])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 1000.0**2, 100.0**2])

dt = 0.01
totalTime = 4 #minutes
IMU = IMUSimulator(dt)
sat_angles = [30, 75, 120, 160, 220]
GPS = GPSR(dt * 100, sat_angles) # second argument is satellite angles in degrees

sigma_accelx_white = 0.04
sigma_accely_white = 0.04
sigma_gyro_white = 0.006

sigma_accelx_walk = 0.001
sigma_accely_walk = 0.001
sigma_gyro_walk = 0.0001

# Standard White Noise Variances (from the tau=1 intercept)
var_vx     = (sigma_accelx_white ** 2) * dt
var_vy     = (sigma_accely_white ** 2) * dt
var_theta = (sigma_gyro_white ** 2) * dt

# Bias Drift Variances (from the sloped right side of the Allan plot)
var_bax_walk = (sigma_accelx_walk ** 2) * dt
var_bay_walk = (sigma_accely_walk ** 2) * dt
var_bw_walk = (sigma_gyro_walk ** 2) * dt

sigma_clk_walk = 0.1

Q = np.diag([
    0.0, 0.0, 
    (sigma_accelx_white**2) * dt, 
    (sigma_accely_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accelx_walk**2)  * dt, 
    (sigma_accely_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt,
    (sigma_clk_walk**2) * dt,
    (0.05**2) * dt
])

Rpr = np.eye(len(sat_angles), dtype = float) * 2.0**2
Rdr = np.eye(len(sat_angles), dtype = float) * 0.05**2
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
eskf_path, = ax1.plot([], [], 'm--', label='EKF')
ax1.legend()

history_eskf_x, history_eskf_y = [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-0.5, 0.5)   # Error in meters
ax2.set_title("DR Residuals")
ax2.set_ylabel("Error (m)")
ax2.grid(True)

ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim(-5, 5)   # Error in meters
ax3.set_title("PR Residuals")
ax3.set_ylabel("Error (m)")
ax3.grid(True)

# link a specific Satellite ID (PRN) to its data history and its plot line.
history_time = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_respr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectsdr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
line_objectspr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_resdr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin( omega * t)
    
    ax, ay, omegahat, trueGyroBias, trueAccelXBias, trueAccelYBias = IMU.generate_measurements(true_ax_body = 0, true_ay_body = -omega**2 * radius, true_omega = omega)
    ax_corr = ax - error_states[5] # raw_accelx - b_ax
    ay_corr = ay - error_states[6] # raw_accely - b_ay
    w_corr = omegahat - error_states[7] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward
    x, y, vx, vy, theta, bax, bay, bw, bclk, drift_clk = x_hat
    
    # 2. Rotate Body-Frame accelerations into the Global Earth-Frame
    # This uses the standard 2D rotation matrix:
    # [ cos(theta)  -sin(theta) ]
    # [ sin(theta)   cos(theta) ]
    a_global_x = (ax_corr * np.cos(theta)) - (ay_corr * np.sin(theta))
    a_global_y = (ax_corr * np.sin(theta)) + (ay_corr * np.cos(theta))
    vx     += a_global_x * dt
    vy     += a_global_y * dt
    theta += w_corr * dt
    x     += vx * dt
    y     += vy * dt
    bclk  += drift_clk * dt
    x_hat = np.array([x, y, vx, vy, theta, bax, bay, bw, bclk, drift_clk])

    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = np.eye(10)
    F[0, 2] = dt #dx/dvx
    F[1, 3] = dt #dy/dvy
    F[2, 4] = -a_global_y * dt # dvx/dtheta
    F[3, 4] = a_global_x * dt #dvy/dtheta
    F[2, 5] =  -np.cos(theta) * dt #dvx/db_ax
    F[2, 6] =  np.sin(theta) * dt #dvx/db_ay
    F[3, 5] =  -np.sin(theta) * dt #dvy/db_ax
    F[3, 6] =  -np.cos(theta) * dt #dvy/db_ay
    F[4, 7] = -dt #dtheta/db_w
    F[8, 9] = dt # dbclk/ddrift_clk

    # Update Covariance using the Jacobian
    P = F @ P @ F.T + Q

    active_prns=[0,1,2,3,4] #includes 0 as a PRN

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, -omega * radius * np.sin(omega * t), omega * radius * np.cos(omega * t) )
        residualpr = np.zeros(len(sat_angles))
        residualdr = np.zeros(len(sat_angles))
        Hpr = np.zeros((len(sat_angles), 10))
        Hdr = np.zeros((len(sat_angles), 10))
        HprSerial = np.zeros((1,10))
        HdrSerial = np.zeros((1,10))
        sat_x = []
        sat_y = []
   
        if t < 5:
            #batch incorporation of measurements
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    Hpr[j][0] = ux
                    Hpr[j][1] = uy
                    Hpr[j][8] = 1.0
                    
                    pr_hat = r_nom + x_hat[8]
                    ux*=28
                    uy*=28
                    residualpr[j] = rawPRs[j] - pr_hat
                    
                    if j not in history_time:
                        history_time[j] = []
                        history_respr[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax3.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectspr[j] = new_line
                        ax3.legend(loc='upper right') # Update legend with new sat
                        history_resdr[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax2.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectsdr[j] = new_line
                        ax2.legend(loc='upper right') # Update legend with new sat
                    
                    # Append the current time and residual to this specific satellite's history
                    history_time[j].append(t)
                    history_respr[j].append(residualpr[j])
                    
                    # Update the specific line object memory pointers
                    line_objectspr[j].set_data(history_time[j], history_respr[j])
                    
                    sat_x.append(ux)
                    sat_y.append(uy)
                    
            sat_plot.set_data(sat_x, sat_y)
            K = P @ Hpr.T @ np.linalg.inv (Hpr @ P @ Hpr.T + Rpr)
            error_states = K @ residualpr.T
            P = (np.eye(10) -  K @ Hpr) @ P
            P = 0.5 * (P + P.T)

            # --- INJECTION STEP ---
            # Apply error estimations directly to nominal totals
            x_hat[0] += error_states[0]  # Fix X
            x_hat[ 1] += error_states[1]  # Fix Y
            x_hat[ 2] += error_states[2]  # Fix Velocity
            x_hat[ 3] += error_states[3]  # Fix Heading
            x_hat[ 4] += error_states[4]
            x_hat[ 5] += error_states[5]  
            x_hat[ 6] += error_states[6]
            x_hat[ 7] += error_states[7]
            x_hat[ 8] += error_states[8]
            x_hat[ 9] += error_states[9]
            
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    Hdr[j][2] = ux
                    Hdr[j][3] = uy 
                    Hdr[j][9] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[9]
                    residualdr[j] = rawDRs[j] - dr_hat
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdr[j])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
            
            K = P @ Hdr.T @ np.linalg.inv (Hdr @ P @ Hdr.T + Rdr)
            error_states = K @ residualdr.T
            P = (np.eye(10) -  K @ Hdr) @ P
            P = 0.5 * (P + P.T)

            x_hat[0] += error_states[0]  # Fix X
            x_hat[ 1] += error_states[1]  # Fix Y
            x_hat[ 2] += error_states[2]  # Fix Velocity
            x_hat[ 3] += error_states[3]  # Fix Heading
            x_hat[ 4] += error_states[4]
            x_hat[ 5] += error_states[5]  
            x_hat[ 6] += error_states[6]
            x_hat[ 7] += error_states[7]
            x_hat[ 8] += error_states[8]
            x_hat[ 9] += error_states[9]
        
        else:
            #serial incorporation of measurements
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    HprSerial[0][0] = ux
                    HprSerial[0][1] = uy
                    HprSerial[0][8] = 1.0
                    
                    pr_hat = r_nom + x_hat[8]
                    ux*=28
                    uy*=28
                    residualprSerial = np.array([rawPRs[j] - pr_hat])
                    
                    if j not in history_time:
                        history_time[j] = []
                        history_respr[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax3.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectspr[j] = new_line
                        ax3.legend(loc='upper right') # Update legend with new sat
                        history_resdr[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax2.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectsdr[j] = new_line
                        ax2.legend(loc='upper right') # Update legend with new sat
                    
                    # Append the current time and residual to this specific satellite's history
                    history_time[j].append(t)
                    history_respr[j].append(residualprSerial[0])
                    
                    # Update the specific line object memory pointers
                    line_objectspr[j].set_data(history_time[j], history_respr[j])
                    
                    S_inv = 1.0 / (HprSerial @ P @ HprSerial.T + Rpr[0][0])
                    K = P @ HprSerial.T @ S_inv
                    
                    error_states = K @ residualprSerial
                    P = (np.eye(10) -  K @ HprSerial) @ P
                    P = 0.5 * (P + P.T)

                    # --- INJECTION STEP ---
                    # Apply error estimations directly to nominal totals
                    x_hat[0] += error_states[0]  # Fix X
                    x_hat[ 1] += error_states[1]  # Fix Y
                    x_hat[ 2] += error_states[2]  # Fix Velocity
                    x_hat[ 3] += error_states[3]  # Fix Heading
                    x_hat[ 4] += error_states[4]
                    x_hat[ 5] += error_states[5]  
                    x_hat[ 6] += error_states[6]
                    x_hat[ 7] += error_states[7]
                    x_hat[ 8] += error_states[8]
                    x_hat[ 9] += error_states[9]
                
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    HdrSerial[0][2] = ux
                    HdrSerial[0][3] = uy
                    HdrSerial[0][9] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[9]
                    residualdrSerial = np.array([rawDRs[j] - dr_hat])
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdrSerial[0])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
                    S_inv = 1.0 / (HdrSerial @ P @ HdrSerial.T + Rdr[0][0])
                    K = P @ HdrSerial.T @ S_inv
                    error_states = K @ residualdrSerial
                    
                    P = (np.eye(10) -  K @ HdrSerial) @ P
                    P = 0.5 * (P + P.T)

                    x_hat[0] += error_states[0]  # Fix X
                    x_hat[ 1] += error_states[1]  # Fix Y
                    x_hat[ 2] += error_states[2]  # Fix Velocity
                    x_hat[ 3] += error_states[3]  # Fix Heading
                    x_hat[ 4] += error_states[4]
                    x_hat[ 5] += error_states[5]  
                    x_hat[ 6] += error_states[6]
                    x_hat[ 7] += error_states[7]
                    x_hat[ 8] += error_states[8]
                    x_hat[ 9] += error_states[9]
        

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        eskf_path.set_data(history_eskf_x, history_eskf_y)

    #plt.pause(0.0001)
plt.ioff(); plt.show()
