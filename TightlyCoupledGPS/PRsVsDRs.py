import numpy as np
import matplotlib.pyplot as plt
from IMUv1 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5

# --- EKF Initialization ---
x_hat = np.array([radius, 0.0, radius * omega, np.pi/2, 0.0, 0.0, 45.0])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
P = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1000.0**2])

x_hatdr = np.array([radius, 0.0, radius * omega, np.pi/2, 0.0, 0.0, 45.0, 0.14])  
error_statesdr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
Pdr = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1000.0**2, 100.0**2])

dt = 0.01
totalTime = 3 #minutes
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

# Noise Covariances
Q = np.diag([
    0.0, 0.0, 
    (sigma_accel_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accel_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt,
    (sigma_clk_walk**2) * dt
])

Qdr = np.diag([
    0.0, 0.0, 
    (sigma_accel_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accel_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt,
    (sigma_clk_walk**2) * dt,
    (0.05**2) * dt
])

R = np.eye(len(sat_angles), dtype = float) * 2.0**2
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
eskf_path, = ax1.plot([], [], 'b--', label='ESKF')
eskfdr_path, = ax1.plot([], [], 'm--', label='ESKFDR')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_eskfdr_x, history_eskfdr_y = [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("PR Residuals No DR")
ax2.set_ylabel("Error (m)")
ax2.grid(True)

ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax3.set_title("PR Residuals With DR")
ax3.set_ylabel("Error (m)")
ax3.grid(True)

# link a specific Satellite ID (PRN) to its data history and its plot line.
history_time = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_res = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objects = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_resdr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectsdr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }

I7 = np.eye(7)
I8 = np.eye(8)
residual = np.zeros(len(sat_angles))
H = np.zeros((len(sat_angles), 7))
residualdr = np.zeros(len(sat_angles))
dr_residual = np.zeros(len(sat_angles))
Hdr = np.zeros((len(sat_angles), 8))
drH = np.zeros((len(sat_angles), 8))
for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    a, omegahat, trueGyroBias, trueAccelBias = IMU.generate_measurements(true_a_body = 0, true_omega = omega)
    a_corr = a - x_hat[4] # raw_accel - b_a
    w_corr = omegahat - x_hat[5] # raw_gyro - b_w
    a_corrdr = a - x_hatdr[4] # raw_accel - b_a
    w_corrdr = omegahat - x_hatdr[5] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward using trig
    x, y, v, theta, ba, bw, bclk = x_hat
    xdr, ydr, vdr, thetadr, badr, bwdr, bclkdr, clkwdr = x_hatdr
    
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    v     += a_corr * dt
    theta += w_corr * dt
    x_hat = np.array([x, y, v, theta, ba, bw, bclk])
    
    xdr     += vdr * np.cos(thetadr) * dt
    ydr     += vdr * np.sin(thetadr) * dt
    vdr     += a_corrdr * dt
    thetadr += w_corrdr * dt
    bclkdr  += clkwdr * dt
    x_hatdr = np.array([xdr, ydr, vdr, thetadr, badr, bwdr, bclkdr, clkwdr])

    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = I7.copy()
    F[0, 2] = np.cos(theta) * dt
    F[0, 3] = -v * np.sin(theta) * dt
    F[1, 2] = np.sin(theta) * dt
    F[1, 3] = v * np.cos(theta) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt
    
    Fdr = I8.copy()
    Fdr[0, 2] = np.cos(thetadr) * dt
    Fdr[0, 3] = -vdr * np.sin(thetadr) * dt
    Fdr[1, 2] = np.sin(thetadr) * dt
    Fdr[1, 3] = vdr * np.cos(thetadr) * dt
    Fdr[2, 4] = -dt
    Fdr[3, 5] = -dt
    Fdr[6, 7] = dt

    # Update Covariance using the Jacobian
    P = F @ P @ F.T + Q
    Pdr = Fdr @ Pdr @ Fdr.T + Qdr

    active_prns=[0,1,2,3,4] #includes 0 as a PRN

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, -omega * radius * np.sin(omega * t), omega * radius * np.cos(omega * t) )
        residual.fill(0)
        H.fill(0)
        residualdr.fill(0)
        dr_residual.fill(0)
        Hdr.fill(0)
        drH.fill(0)
        sat_x = []
        sat_y = []
    
        #batch incorporation of measurements
        for j in range(len(estimated_sat_pos)):
            if j in active_prns:
                dx = x_hat[0] - estimated_sat_pos[j][0]
                dy = x_hat[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                H[j][0] = ux
                H[j][1] = uy
                H[j][6] = 1.0
                
                pr_hat = r_nom + x_hat[6]
                residual[j] = rawPRs[j] - pr_hat

                dx = x_hatdr[0] - estimated_sat_pos[j][0]
                dy = x_hatdr[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                Hdr[j][0] = ux
                Hdr[j][1] = uy
                Hdr[j][6] = 1.0
                
                pr_hat = r_nom + x_hatdr[6]
                ux*=28
                uy*=28
                residualdr[j] = rawPRs[j] - pr_hat
                
                if j not in history_time:
                    history_time[j] = []
                    history_res[j] = []
                    history_resdr[j] = []
                    # Create a new line with markers to see individual updates clearly
                    new_line, = ax2.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                    line_objects[j] = new_line
                    ax2.legend(loc='upper right') # Update legend with new sat
                    new_line, = ax3.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                    line_objectsdr[j] = new_line
                    ax3.legend(loc='upper right') # Update legend with new sat
                
                
                # Append the current time and residual to this specific satellite's history
                history_time[j].append(t)
                history_res[j].append(residual[j])
                history_resdr[j].append(residualdr[j])
                
                # Update the specific line object memory pointers
                line_objects[j].set_data(history_time[j], history_res[j])
                line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                
                sat_x.append(ux)
                sat_y.append(uy)
                
        sat_plot.set_data(sat_x, sat_y)
        K = P @ H.T @ np.linalg.inv (H @ P @ H.T + R)
        error_states = K @ residual.T
        P = (I7 -  K @ H) @ P
        P = 0.5 * (P + P.T)
        
        Kdr = Pdr @ Hdr.T @ np.linalg.inv (Hdr @ Pdr @ Hdr.T + R)
        error_statesdr = Kdr @ residualdr.T
        Pdr = (I8 -  Kdr @ Hdr) @ Pdr
        Pdr = 0.5 * (Pdr + Pdr.T)

        # --- INJECTION STEP ---
        # Apply error estimations directly to nominal totals
        x_hat[0] += error_states[0]  # Fix X
        x_hat[ 1] += error_states[1]  # Fix Y
        x_hat[ 2] += error_states[2]  # Fix Velocity
        x_hat[ 3] += error_states[3]  # Fix Heading
        x_hat[ 4] += error_states[4]
        x_hat[ 5] += error_states[5]  
        x_hat[ 6] += error_states[6]
        
        x_hatdr[0] += error_statesdr[0]  # Fix X
        x_hatdr[ 1] += error_statesdr[1]  # Fix Y
        x_hatdr[ 2] += error_statesdr[2]  # Fix Velocity
        x_hatdr[ 3] += error_statesdr[3]  # Fix Heading
        x_hatdr[ 4] += error_statesdr[4]
        x_hatdr[ 5] += error_statesdr[5]  
        x_hatdr[ 6] += error_statesdr[6]
        x_hatdr[ 7] += error_statesdr[7]
        
        for j in range(len(estimated_sat_pos)):
            if j in active_prns:
                
                dx = x_hatdr[0] - estimated_sat_pos[j][0]
                dy = x_hatdr[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                
                drH[j][2] = ux * np.cos(x_hatdr[3]) + uy * np.sin(x_hatdr[3])
                drH[j][3] = ux * (-x_hatdr[2] * np.sin(x_hatdr[3])) + uy * (x_hatdr[2] * np.cos(x_hatdr[3]))
                drH[j][7] = 1.0

                # Expected relative velocity
                #these sat vels are only valid for a circular flight path
                satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                vx = x_hatdr[2] * np.cos(x_hatdr[3]) - satvelX
                vy = x_hatdr[2] * np.sin(x_hatdr[3]) - satvelY
                dr_hat = (ux * vx) + (uy * vy) + x_hatdr[7]
                dr_residual[j] = rawDRs[j] - dr_hat
        
        Kdr = Pdr @ drH.T @ np.linalg.inv (drH @ Pdr @ drH.T + Rdr)
        error_statesdr = Kdr @ dr_residual.T
        Pdr = (I8 -  Kdr @ drH) @ Pdr
        Pdr = 0.5 * (Pdr + Pdr.T)

        x_hatdr[0] += error_statesdr[0]  # Fix X
        x_hatdr[ 1] += error_statesdr[1]  # Fix Y
        x_hatdr[ 2] += error_statesdr[2]  # Fix Velocity
        x_hatdr[ 3] += error_statesdr[3]  # Fix Heading
        x_hatdr[ 4] += error_statesdr[4]
        x_hatdr[ 5] += error_statesdr[5]  
        x_hatdr[ 6] += error_statesdr[6]
        x_hatdr[ 7] += error_statesdr[7]

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])
    
    history_eskfdr_x.append(x_hatdr[0])
    history_eskfdr_y.append(x_hatdr[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        eskf_path.set_data(history_eskf_x, history_eskf_y)
        eskfdr_path.set_data(history_eskfdr_x, history_eskfdr_y)
        

    #plt.pause(0.0001)
plt.ioff(); plt.show()
