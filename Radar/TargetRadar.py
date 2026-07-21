import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv3 import IMUSimulator
from GPSv1 import GPSR
from Radarv2 import RadarSimulator

radius = 20
omega = 0.5
tgtInitPosX = 0
tgtInitPosY = 1000
tgtVelX = -10
tgtVelY = 0

#state: x, y, vx, vy, theta, b_ax, b_ay, S_x, S_y, M_xy b_w, b_clk, b_drift, tgtx, tgty, tgtvelx, tgtvely
x_hat = np.array([0.0, 0.0, radius * omega, 2 * radius * omega, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 0.14, 0.0, 1000.0, 0.0, 0.0])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 4e-4, 4e-4, 4e-4, 1000.0**2, 100.0**2, 1000.0**2, 1000.0**2, 10.0**2, 10.0**2])

dt = 0.01
totalTime = 8 #minutes
IMU = IMUSimulator(dt)
sat_angles = [30, 75, 120, 160]
GPS = GPSR(dt * 100, sat_angles) # second argument is satellite angles in degrees
Radar = RadarSimulator()

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
sigma_sf_walk = 0.00001

sigma_target_accel = 5.0 # Max expected acceleration in m/s^2

Q = np.diag([
    0.0, 0.0, 
    (sigma_accelx_white**2) * dt, 
    (sigma_accely_white**2) * dt, 
    (sigma_gyro_white**2) * dt,
    (sigma_accelx_walk**2)  * dt, 
    (sigma_accely_walk**2)  * dt, 
    (sigma_gyro_walk**2)  * dt,
    (sigma_sf_walk**2) * dt, 
    (sigma_sf_walk**2) * dt, 
    (sigma_sf_walk**2) * dt, 
    (sigma_clk_walk**2) * dt,
    (0.05**2) * dt,
    0.0,
    0.0,
    (sigma_target_accel**2) * dt**2,
    (sigma_target_accel**2) * dt**2
])

Rpr = np.eye(len(sat_angles), dtype = float) * 2.0**2
Rdr = np.eye(len(sat_angles), dtype = float) * 0.05**2
R_radar = np.diag([2.0**2, np.radians(0.5)**2, 0.1**2]) # R, Phi, RR variances

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
drone_dot, = ax1.plot([], [], 'go', label='Truth')
target_dot, = ax1.plot([], [], 'bo', label='Target')
sat_plot, = ax1.plot([], [], 'ro', markersize=8, label="Visible Sats")
eskf_path, = ax1.plot([], [], 'm--', label='EKF')
ax1.legend()

history_eskf_x, history_eskf_y = [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-1, 1)   # Error in meters
ax2.set_title("DR Residuals")
ax2.set_ylabel("Error (m)")
ax2.grid(True)

ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim(-5, 5)   # Error in meters
ax3.set_title("PR Residuals")
ax3.set_ylabel("Error (m)")
ax3.grid(True)

# Bottom Plot: Residuals (z - Hx)
ax4.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax4.set_ylim(-6, 6)   # Error in meters
ax4.set_title("Radar residuals")
ax4.set_ylabel("Error (m)")
range_line, = ax4.plot([], [], 'b-', label=' Range Residual', alpha=0.6)
bearing_line, = ax4.plot([], [], 'g-', label=' Bearing Residual', alpha=0.6)
rangerate_line, = ax4.plot([], [], 'r-', label=' Range Rate Residual', alpha=0.6)
ax4.legend()
ax4.grid(True)

# link a specific Satellite ID (PRN) to its data history and its plot line.
history_time = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_time_radar = []   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_respr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectspr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_resdr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectsdr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_res_radarrange = []    
history_res_radarbearing = []  
history_res_radarrate = []   

spin_rate = omega * 2.66

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.sin(omega * t)
    curr_y = radius * np.sin(2 * omega * t)
    curr_x_tgt = tgtInitPosX + tgtVelX * t
    curr_y_tgt = tgtInitPosY + tgtVelY * t
    curr_theta = np.pi / 2 + t * spin_rate
    curr_velx = radius * omega * np.cos(omega * t)
    curr_vely = 2 * radius * omega * np.cos(2 * omega * t)
    curr_accx = -radius * omega**2 * np.sin(omega * t)
    curr_accy = -radius * (2 * omega)**2 * np.sin(2 * omega * t)
    body_ax = (curr_accx * np.cos(curr_theta)) + (curr_accy * np.sin(curr_theta))
    body_ay = -(curr_accx * np.sin(curr_theta)) + (curr_accy * np.cos(curr_theta))
    
    ax, ay, omegahat, trueGyroBias, trueAccelXBias, trueAccelYBias, true_Sx, true_Sy, true_Mxy = IMU.generate_measurements(true_ax_body = body_ax, true_ay_body = body_ay, true_omega = spin_rate)
    
    ax_corr = ax - ax * x_hat[8] - ay * x_hat[10] - x_hat[5]
    ay_corr = ay - ay * x_hat[9] - ax * x_hat[10] - x_hat[6] 
    w_corr = omegahat - x_hat[7] # raw_gyro - b_w
    
    # 2. EKF PREDICT: Move the state forward
    x, y, vx, vy, theta, bax, bay, bw, Sx, Sy, Mxy, bclk, drift_clk, tgtx, tgty, tgtvx, tgtvy = x_hat
    
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
    tgtx  += tgtvx * dt
    tgty  += tgtvy * dt
    x_hat = np.array([x, y, vx, vy, theta, bax, bay, bw, Sx, Sy, Mxy, bclk, drift_clk, tgtx, tgty, tgtvx, tgtvy])
    
    # 2. LINEARIZE
    F = np.eye(17)
    F[0, 2] = dt #dx/dvx
    F[1, 3] = dt #dy/dvy
    F[2, 4] = -a_global_y * dt # dvx/dtheta
    F[3, 4] = a_global_x * dt #dvy/dtheta
    F[2, 5] =  -np.cos(theta) * dt #dvx/db_ax
    F[2, 6] =  np.sin(theta) * dt #dvx/db_ay
    F[2, 8] =  -ax * np.cos(theta) * dt #dvx/dSx
    F[2, 9] =  ay * np.sin(theta) * dt #dvx/dSy
    F[2, 10] =  -ay * np.cos(theta) * dt + ax * np.sin(theta) * dt #dvx/dMxy
    F[3, 5] =  -np.sin(theta) * dt #dvy/db_ax
    F[3, 6] =  -np.cos(theta) * dt #dvy/db_ay
    F[3, 8] =  -ax * np.sin(theta) * dt #dvy/dSx
    F[3, 9] =  -ay * np.cos(theta) * dt #dvy/dSy
    F[3, 10] =  -ay * np.sin(theta) * dt - ax * np.cos(theta) * dt #dvy/dMxy
    F[4, 7] = -dt #dtheta/db_w
    F[11, 12] = dt # dbclk/ddrift_clk
    F[13, 15] = dt # dx_t/dvx_t
    F[14, 16] = dt # dy_t/dvy_t

    # Update Covariance using the Jacobian
    P = F @ P @ F.T + Q

    active_prns=[0,1,2,3] #includes 0 as a PRN
    if t > 120:
        active_prns = []

    #radar updates
    if i % int(0.1/dt) == 0:
        # 1. Get noisy measurement from hardware
        raw_r, raw_phi, raw_rr = Radar.get_measurements(curr_x, curr_y, curr_velx, curr_vely, curr_x_tgt, curr_y_tgt, tgtVelX, tgtVelY)
        z_radar = np.array([raw_r, raw_phi, raw_rr])
        
        dx = x_hat[13] - x_hat[0]
        dy = x_hat[14] - x_hat[1]
        r_nom = np.sqrt(dx**2 + dy**2)
        if r_nom < 1e-6: r_nom = 1e-6
        ux = dx / r_nom
        uy = dy / r_nom
        
        r_hat = r_nom
        phi_hat = np.arctan2(dy, dx)
        rr_hat = (ux * (x_hat[15] - x_hat[2])) + (uy * (x_hat[16] - x_hat[3]))
        
        z_hat = np.array([r_hat, phi_hat, rr_hat])
        res_radar = z_radar - z_hat
        res_radar[1] = (res_radar[1] + np.pi) % (2 * np.pi) - np.pi # Angle wrap!
        
        history_time_radar.append(t)
        history_res_radarrange.append(res_radar[0])
        history_res_radarbearing.append(res_radar[1])  
        history_res_radarrate.append(res_radar[2])
        
        range_line.set_data(history_time_radar, history_res_radarrange)
        bearing_line.set_data(history_time_radar, history_res_radarbearing)
        rangerate_line.set_data(history_time_radar, history_res_radarrate)

        H_radar = np.zeros((3, 17))
        
        #range derivatives
        H_radar[0, 0] = -dx / r_nom  # w.r.t Drone X
        H_radar[0, 1] = -dy / r_nom  # w.r.t Drone Y
        H_radar[0, 13] = dx / r_nom  # w.r.t Target X
        H_radar[0, 14] = dy / r_nom  # w.r.t Target Y

        # --- Bearing Derivatives ---
        H_radar[1, 0] = dy / (r_nom**2)   # w.r.t Drone X
        H_radar[1, 1] = -dx / (r_nom**2)  # w.r.t Drone Y
        H_radar[1, 13] = -dy / (r_nom**2) # w.r.t Target X
        H_radar[1, 14] = dx / (r_nom**2)  # w.r.t Target Y

        #range rate derivatives
        v_rel_x = x_hat[15] - x_hat[2]
        v_rel_y = x_hat[16] - x_hat[3]
        H_radar[2, 0] = -(v_rel_x - ux * rr_hat) / r_nom  # w.r.t. Drone X
        H_radar[2, 1] = -(v_rel_y - uy * rr_hat) / r_nom  # w.r.t. Drone Y
        H_radar[2, 13] = (v_rel_x - ux * rr_hat) / r_nom  # w.r.t. Drone X
        H_radar[2, 14] = (v_rel_y - uy * rr_hat) / r_nom  # w.r.t. Drone Y
        H_radar[2, 2] = -ux;
        H_radar[2, 3] = -uy;
        H_radar[2, 15] = ux;
        H_radar[2, 16] = uy;

        S_inv = np.linalg.inv(H_radar @ P @ H_radar.T + R_radar)
        K = P @ H_radar.T @ S_inv
        error_states = K @ res_radar
        P = (np.eye(17) - K @ H_radar) @ P
        P = 0.5 * (P + P.T)
        
        for idx in range(17): x_hat[idx] += error_states[idx]

    # GPS KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, curr_velx, curr_vely )
        residualpr = np.zeros(len(sat_angles))
        residualdr = np.zeros(len(sat_angles))
        
        Hpr = np.zeros((len(sat_angles), 17))
        Hdr = np.zeros((len(sat_angles), 17))
        HprSerial = np.zeros((1,17))
        HdrSerial = np.zeros((1,17))
        
        sat_x = []
        sat_y = []
   
        if t < 1 * 5:
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
                    Hpr[j][11] = 1.0
                    
                    pr_hat = r_nom + x_hat[11]
                    residualpr[j] = rawPRs[j] - pr_hat
                    ux*=28
                    uy*=28
                    
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
            P = (np.eye(17) -  K @ Hpr) @ P
            P = 0.5 * (P + P.T)

            for j in range(17):
                x_hat[j] += error_states[j] 

            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    Hdr[j][2] = ux
                    Hdr[j][3] = uy 
                    Hdr[j][12] = 1.0
                    
                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[12]
                    residualdr[j] = rawDRs[j] - dr_hat
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdr[j])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
            
            K = P @ Hdr.T @ np.linalg.inv (Hdr @ P @ Hdr.T + Rdr)
            error_states = K @ residualdr.T
            P = (np.eye(17) -  K @ Hdr) @ P
            P = 0.5 * (P + P.T)
            
            for j in range(17):
                x_hat[j] += error_states[j] 
        
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
                    HprSerial[0][11] = 1.0
                    
                    pr_hat = r_nom + x_hat[11]
                    residualprSerial = np.array([rawPRs[j] - pr_hat])
                    ux*=28
                    uy*=28
                    
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
                    P = (np.eye(17) -  K @ HprSerial) @ P
                    P = 0.5 * (P + P.T)

                    for k in range(17):
                        x_hat[k] += error_states[k] 
                

            for j in range(len(estimated_sat_pos)):
                if j in active_prns:
                    
                    dx = x_hat[0] - estimated_sat_pos[j][0]
                    dy = x_hat[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    HdrSerial[0][2] = ux
                    HdrSerial[0][3] = uy
                    HdrSerial[0][12] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[12]
                    residualdrSerial = np.array([rawDRs[j] - dr_hat])
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdrSerial[0])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    
                    S_inv = 1.0 / (HdrSerial @ P @ HdrSerial.T + Rdr[0][0])
                    K = P @ HdrSerial.T @ S_inv
                    error_states = K @ residualdrSerial
                    P = (np.eye(17) -  K @ HdrSerial) @ P
                    P = 0.5 * (P + P.T)
                    
                    for k in range(17):
                        x_hat[k] += error_states[k] 
        

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        target_dot.set_data([curr_x_tgt/ 100], [curr_y_tgt/ 100])
        eskf_path.set_data(history_eskf_x, history_eskf_y)

#    plt.pause(0.0001)
plt.ioff(); plt.show()
