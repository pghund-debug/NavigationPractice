import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5

x_hat = np.array([radius, 0.0, radius * omega, np.pi/2, 0.0, 0.0, 45.0, 0.14])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
P = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1000.0**2, 100.0**2])

dt = 0.01
totalTime = 4 #minutes
IMU = IMUSimulator(dt)
sat_angles = [30, 75, 120, 160, 220]
GPS = GPSR(dt * 100, sat_angles) # second argument is satellite angles in degrees

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

Q = np.diag([
    0.0, 0.0, 
    (sigma_accel_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accel_walk**2)  * dt, 
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
eskf_path, = ax1.plot([], [], 'm--', label='ESKFDR')
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
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corr = a - error_states[4] # raw_accel - b_a
    w_corr = omegahat - error_states[5] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    x, y, v, theta, ba, bw, bclk, clkw = x_hat
    
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    v     += a_corr * dt
    theta += w_corr * dt
    bclk  += clkw * dt
    x_hat = np.array([x, y, v, theta, ba, bw, bclk, clkw])

    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = np.eye(8)
    F[0, 2] = np.cos(theta) * dt
    F[0, 3] = -v * np.sin(theta) * dt
    F[1, 2] = np.sin(theta) * dt
    F[1, 3] = v * np.cos(theta) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt
    F[6, 7] = dt

    # Update Covariance using the Jacobian
    P = F @ P @ F.T + Q

    if t < 150:
        active_prns=[0,1,2,3,4] #includes 0 as a PRN
    if t < 135:
        active_prns=[0,1,2,3,4] #includes 0 as a PRN
    if t < 120:
        active_prns=[0] #includes 0 as a PRN
    if t < 105:
        active_prns=[0] #includes 0 as a PRN
    if t < 90:
        active_prns=[0] #includes 0 as a PRN
    if t < 60:
        active_prns=[0] #includes 0 as a PRN
    if t < 30:
        active_prns=[0] #includes 0 as a PRN
    if t < 15:
        active_prns=[0,1,2,3,4] #includes 0 as a PRN
    if t < 10:
        active_prns=[0,1,2,3,4] #includes 0 as a PRN
    

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, -omega * radius * np.sin(omega * t), omega * radius * np.cos(omega * t) )
        residualpr = np.zeros(len(sat_angles))
        residualdr = np.zeros(len(sat_angles))
        Hpr = np.zeros((len(sat_angles), 8))
        Hdr = np.zeros((len(sat_angles), 8))
        HprSerial = np.zeros((1,8))
        HdrSerial = np.zeros((1,8))
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
                    Hpr[j][6] = 1.0
                    
                    pr_hat = r_nom + x_hat[6]
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
            P = (np.eye(8) -  K @ Hpr) @ P
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
            
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    Hdr[j][2] = ux * np.cos(x_hat[3]) + uy * np.sin(x_hat[3])
                    Hdr[j][3] = ux * (-x_hat[2] * np.sin(x_hat[3])) + uy * (x_hat[2] * np.cos(x_hat[3]))
                    Hdr[j][7] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    vx = x_hat[2] * np.cos(x_hat[3]) - satvelX
                    vy = x_hat[2] * np.sin(x_hat[3]) - satvelY
                    dr_hat = (ux * vx) + (uy * vy) + x_hat[7]
                    residualdr[j] = rawDRs[j] - dr_hat
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdr[j])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
            
            K = P @ Hdr.T @ np.linalg.inv (Hdr @ P @ Hdr.T + Rdr)
            error_states = K @ residualdr.T
            P = (np.eye(8) -  K @ Hdr) @ P
            P = 0.5 * (P + P.T)

            x_hat[0] += error_states[0]  # Fix X
            x_hat[ 1] += error_states[1]  # Fix Y
            x_hat[ 2] += error_states[2]  # Fix Velocity
            x_hat[ 3] += error_states[3]  # Fix Heading
            x_hat[ 4] += error_states[4]
            x_hat[ 5] += error_states[5]  
            x_hat[ 6] += error_states[6]
            x_hat[ 7] += error_states[7]
        
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
                    HprSerial[0][6] = 1.0
                    
                    pr_hat = r_nom + x_hat[6]
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
                    P = (np.eye(8) -  K @ HprSerial) @ P
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
                
            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    HdrSerial[0][2] = ux * np.cos(x_hat[3]) + uy * np.sin(x_hat[3])
                    HdrSerial[0][3] = ux * (-x_hat[2] * np.sin(x_hat[3])) + uy * (x_hat[2] * np.cos(x_hat[3]))
                    HdrSerial[0][7] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    vx = x_hat[2] * np.cos(x_hat[3]) - satvelX
                    vy = x_hat[2] * np.sin(x_hat[3]) - satvelY
                    dr_hat = (ux * vx) + (uy * vy) + x_hat[7]
                    residualdrSerial = np.array([rawDRs[j] - dr_hat])
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdrSerial[0])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
                    S_inv = 1.0 / (HdrSerial @ P @ HdrSerial.T + Rdr[0][0])
                    K = P @ HdrSerial.T @ S_inv
                    error_states = K @ residualdrSerial
                    
                    P = (np.eye(8) -  K @ HdrSerial) @ P
                    P = 0.5 * (P + P.T)

                    x_hat[0] += error_states[0]  # Fix X
                    x_hat[ 1] += error_states[1]  # Fix Y
                    x_hat[ 2] += error_states[2]  # Fix Velocity
                    x_hat[ 3] += error_states[3]  # Fix Heading
                    x_hat[ 4] += error_states[4]
                    x_hat[ 5] += error_states[5]  
                    x_hat[ 6] += error_states[6]
                    x_hat[ 7] += error_states[7]
        


    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        eskf_path.set_data(history_eskf_x, history_eskf_y)

    #plt.pause(0.0001)
plt.ioff(); plt.show()
