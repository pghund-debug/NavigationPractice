import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv3 import IMUSimulator
from GPSv1 import GPSR
from Radarv1 import RadarSimulator

radius = 20
omega = 0.5
radar_x = 0
radar_y = 20

#state: x, y, vx, vy, theta, b_ax, b_ay, S_x, S_y, M_xy b_w, b_clk, b_drift
x_hat = np.array([0.0, 0.0, radius * omega, 2 * radius * omega, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 0.14])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 4e-4, 4e-4, 4e-4, 1000.0**2, 100.0**2])

x_hat2 = np.array([0.0, 0.0, radius * omega, 2 * radius * omega, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 0.14])  
error_states2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P2 = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 4e-4, 4e-4, 4e-4, 1000.0**2, 100.0**2])

dt = 0.01
totalTime = 8 #minutes
IMU = IMUSimulator(dt)
sat_angles = [30, 75, 120, 160]
GPS = GPSR(dt * 100, sat_angles) # second argument is satellite angles in degrees
Radar = RadarSimulator(radar_x, radar_y, dt * 10)

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
    (0.05**2) * dt
])

Rpr = np.eye(len(sat_angles), dtype = float) * 2.0**2
Rdr = np.eye(len(sat_angles), dtype = float) * 0.05**2
R_radar = np.diag([2.0**2, np.radians(0.5)**2, 0.1**2]) # R, Phi, RR variances

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
drone_dot, = ax1.plot([], [], 'go', label='Truth')
sat_plot, = ax1.plot([], [], 'ro', markersize=8, label="Visible Sats")
radar_plot, = ax1.plot([radar_x], [radar_y], 'k^', markersize=10, label="Radar")
eskf_path, = ax1.plot([], [], 'm--', label='EKF')
eskf2_path, = ax1.plot([], [], 'c--', label='EKFradar')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_eskf2_x, history_eskf2_y = [], []

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
ax4.set_ylim(-1, 1)   # Error in meters
ax4.set_title("DR Residuals With Radar")
ax4.set_ylabel("Error (m)")
ax4.grid(True)

ax5.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax5.set_ylim(-5, 5)   # Error in meters
ax5.set_title("PR Residuals With Radar")
ax5.set_ylabel("Error (m)")
ax5.grid(True)


# link a specific Satellite ID (PRN) to its data history and its plot line.
history_time = {}   # e.g., { 1: [0, 1, 2...], 3: [0, 1, 2...] }
history_respr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectsdr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
line_objectspr = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_resdr = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }

history_respr2 = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
line_objectsdr2 = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
line_objectspr2 = {}   # e.g., { 1: <matplotlib.lines.Line2D>, 3: <...> }
history_resdr2 = {}    # e.g., { 1: [0.5, 0.4...], 3: [-1.2, -1.5...] }
spin_rate = omega * 2.66

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.sin(omega * t)
    curr_y = radius * np.sin(2 * omega * t)
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
    
    ax_corr2 = ax - ax * x_hat2[8] - ay * x_hat2[10] - x_hat2[5]
    ay_corr2 = ay - ay * x_hat2[9] - ax * x_hat2[10] - x_hat2[6] 
    w_corr2 = omegahat - x_hat2[7] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward
    x, y, vx, vy, theta, bax, bay, bw, Sx, Sy, Mxy, bclk, drift_clk = x_hat
    x2, y2, vx2, vy2, theta2, bax2, bay2, bw2, Sx2, Sy2, Mxy2, bclk2, drift_clk2 = x_hat2
    
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
    x_hat = np.array([x, y, vx, vy, theta, bax, bay, bw, Sx, Sy, Mxy, bclk, drift_clk])

    a_global_x2 = (ax_corr2 * np.cos(theta2)) - (ay_corr2 * np.sin(theta2))
    a_global_y2 = (ax_corr2 * np.sin(theta2)) + (ay_corr2 * np.cos(theta2))
    vx2     += a_global_x2 * dt
    vy2     += a_global_y2 * dt
    theta2 += w_corr2 * dt
    x2     += vx2 * dt
    y2     += vy2 * dt
    bclk2  += drift_clk2 * dt
    x_hat2 = np.array([x2, y2, vx2, vy2, theta2, bax2, bay2, bw2, Sx2, Sy2, Mxy2, bclk2, drift_clk2])

    # 2. LINEARIZE
    F = np.eye(13)
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

    F2 = np.eye(13)
    F2[0, 2] = dt #dx/dvx
    F2[1, 3] = dt #dy/dvy
    F2[2, 4] = -a_global_y2 * dt # dvx/dtheta
    F2[3, 4] = a_global_x2 * dt #dvy/dtheta
    F2[2, 5] =  -np.cos(theta2) * dt #dvx/db_ax
    F2[2, 6] =  np.sin(theta2) * dt #dvx/db_ay
    F2[2, 8] =  -ax * np.cos(theta2) * dt #dvx/dSx
    F2[2, 9] =  ay * np.sin(theta2) * dt #dvx/dSy
    F2[2, 10] =  -ay * np.cos(theta2) * dt + ax * np.sin(theta2) * dt #dvx/dMxy
    F2[3, 5] =  -np.sin(theta2) * dt #dvy/db_ax
    F2[3, 6] =  -np.cos(theta2) * dt #dvy/db_ay
    F2[3, 8] =  -ax * np.sin(theta2) * dt #dvy/dSx
    F2[3, 9] =  -ay * np.cos(theta2) * dt #dvy/dSy
    F2[3, 10] =  -ay * np.sin(theta2) * dt - ax * np.cos(theta2) * dt #dvy/dMxy
    F2[4, 7] = -dt #dtheta/db_w
    F2[11, 12] = dt # dbclk/ddrift_clk

    # Update Covariance using the Jacobian
    P = F @ P @ F.T + Q
    P2 = F2 @ P2 @ F2.T + Q

    active_prns=[0,1,2,3] #includes 0 as a PRN
    if t > 120:
        active_prns = []

    #radar updates
    if i % int(0.1/dt) == 0 and i > 0:
        # 1. Get noisy measurement from hardware
        raw_r, raw_phi, raw_rr = Radar.get_measurements(curr_x, curr_y, curr_velx, curr_vely)
        z_radar = np.array([raw_r, raw_phi, raw_rr])
        
        dx2 = x_hat2[0] - radar_x
        dy2 = x_hat2[1] - radar_y
        r_nom2 = np.sqrt(dx2**2 + dy2**2)
        if r_nom2 < 1e-6: r_nom2 = 1e-6
        ux2 = dx2 / r_nom2
        uy2 = dy2 / r_nom2
        
        r_hat2 = r_nom2
        phi_hat2 = np.arctan2(dy2, dx2)
        rr_hat2 = (ux2 * x_hat2[2]) + (uy2 * x_hat2[3])
        
        z_hat2 = np.array([r_hat2, phi_hat2, rr_hat2])
        res_radar2 = z_radar - z_hat2
        res_radar2[1] = (res_radar2[1] + np.pi) % (2 * np.pi) - np.pi # Angle wrap!
        
        H_radar2 = np.zeros((3, 13))
        H_radar2[0, 0] = ux2; H_radar2[0, 1] = uy2
        H_radar2[1, 0] = -dy2 / (r_nom2**2); H_radar2[1, 1] = dx2 / (r_nom2**2)
        H_radar2[2, 0] = (x_hat2[2] - ux2 * rr_hat2) / r_nom2  # w.r.t. Drone X
        H_radar2[2, 1] = (x_hat2[3] - uy2 * rr_hat2) / r_nom2  # w.r.t. Drone Y
        H_radar2[2, 2] = ux2; H_radar2[2, 3] = uy2
        
        S_inv2 = np.linalg.inv(H_radar2 @ P2 @ H_radar2.T + R_radar)
        K2 = P2 @ H_radar2.T @ S_inv2
        error_states2 = K2 @ res_radar2
        P2 = (np.eye(13) - K2 @ H_radar2) @ P2
        P2 = 0.5 * (P2 + P2.T)
        
        for idx in range(13): x_hat2[idx] += error_states2[idx]


    # GPS KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, curr_velx, curr_vely )
        residualpr = np.zeros(len(sat_angles))
        residualdr = np.zeros(len(sat_angles))
        residualpr2 = np.zeros(len(sat_angles))
        residualdr2 = np.zeros(len(sat_angles))
        
        Hpr = np.zeros((len(sat_angles), 13))
        Hdr = np.zeros((len(sat_angles), 13))
        HprSerial = np.zeros((1,13))
        HdrSerial = np.zeros((1,13))
        
        Hpr2 = np.zeros((len(sat_angles), 13))
        Hdr2 = np.zeros((len(sat_angles), 13))
        HprSerial2 = np.zeros((1,13))
        HdrSerial2 = np.zeros((1,13))
        
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
                    
                    dx = x_hat2[0] - estimated_sat_pos[j][0]
                    dy = x_hat2[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    Hpr2[j][0] = ux
                    Hpr2[j][1] = uy
                    Hpr2[j][11] = 1.0

                    pr_hat = r_nom + x_hat2[11]
                    ux*=28
                    uy*=28
                    residualpr2[j] = rawPRs[j] - pr_hat
                    
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
                        
                        history_respr2[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax5.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectspr2[j] = new_line
                        ax5.legend(loc='upper right') # Update legend with new sat
                        history_resdr2[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax4.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectsdr2[j] = new_line
                        ax4.legend(loc='upper right') # Update legend with new sat
                    

                    # Append the current time and residual to this specific satellite's history
                    history_time[j].append(t)
                    history_respr[j].append(residualpr[j])
                    history_respr2[j].append(residualpr2[j])
                    
                    # Update the specific line object memory pointers
                    line_objectspr[j].set_data(history_time[j], history_respr[j])
                    line_objectspr2[j].set_data(history_time[j], history_respr2[j])
                    
                    sat_x.append(ux)
                    sat_y.append(uy)
                    
            sat_plot.set_data(sat_x, sat_y)
            K = P @ Hpr.T @ np.linalg.inv (Hpr @ P @ Hpr.T + Rpr)
            error_states = K @ residualpr.T
            P = (np.eye(13) -  K @ Hpr) @ P
            P = 0.5 * (P + P.T)
            
            K = P2 @ Hpr2.T @ np.linalg.inv (Hpr2 @ P2 @ Hpr2.T + Rpr)
            error_states2 = K @ residualpr2.T
            P2 = (np.eye(13) -  K @ Hpr2) @ P2
            P2 = 0.5 * (P2 + P2.T)

            for j in range(13):
                x_hat[j] += error_states[j] 
                x_hat2[ j] += error_states2[j]

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
                    
                    dx = x_hat2[0] - estimated_sat_pos[j][0]
                    dy = x_hat2[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    Hdr2[j][2] = ux
                    Hdr2[j][3] = uy 
                    Hdr2[j][12] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[12]
                    dr_hat2 = (ux * (x_hat2[2] - satvelX)) + (uy * (x_hat2[3] - satvelY)) + x_hat2[12]
                    residualdr[j] = rawDRs[j] - dr_hat
                    residualdr2[j] = rawDRs[j] - dr_hat2
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdr[j])
                    history_resdr2[j].append(residualdr2[j])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    line_objectsdr2[j].set_data(history_time[j], history_resdr2[j])
                    
            
            K = P @ Hdr.T @ np.linalg.inv (Hdr @ P @ Hdr.T + Rdr)
            error_states = K @ residualdr.T
            P = (np.eye(13) -  K @ Hdr) @ P
            P = 0.5 * (P + P.T)
            
            K = P2 @ Hdr2.T @ np.linalg.inv (Hdr2 @ P2 @ Hdr2.T + Rdr)
            error_states2 = K @ residualdr2.T
            P2 = (np.eye(13) -  K @ Hdr2) @ P2
            P2 = 0.5 * (P2 + P2.T)

            for j in range(13):
                x_hat[j] += error_states[j] 
                x_hat2[ j] += error_states2[j]
        
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
                    
                    dx = x_hat2[0] - estimated_sat_pos[j][0]
                    dy = x_hat2[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    HprSerial2[0][0] = ux
                    HprSerial2[0][1] = uy
                    HprSerial2[0][11] = 1.0
                    
                    pr_hat = r_nom + x_hat2[11]
                    ux*=28
                    uy*=28
                    residualprSerial2 = np.array([rawPRs[j] - pr_hat])
                    
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
                        
                        history_respr2[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax5.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectspr2[j] = new_line
                        ax5.legend(loc='upper right') # Update legend with new sat
                        history_resdr2[j] = []
                        # Create a new line with markers to see individual updates clearly
                        new_line, = ax4.plot([], [], marker='.', linestyle='-', label=f"PRN {j}")
                        line_objectsdr2[j] = new_line
                        ax4.legend(loc='upper right') # Update legend with new sat
                    
                    # Append the current time and residual to this specific satellite's history
                    history_time[j].append(t)
                    history_respr[j].append(residualprSerial[0])
                    history_respr2[j].append(residualprSerial2[0])
                    
                    # Update the specific line object memory pointers
                    line_objectspr[j].set_data(history_time[j], history_respr[j])
                    line_objectspr2[j].set_data(history_time[j], history_respr2[j])
                    
                    S_inv = 1.0 / (HprSerial @ P @ HprSerial.T + Rpr[0][0])
                    K = P @ HprSerial.T @ S_inv
                    error_states = K @ residualprSerial
                    P = (np.eye(13) -  K @ HprSerial) @ P
                    P = 0.5 * (P + P.T)
                    
                    S_inv = 1.0 / (HprSerial2 @ P2 @ HprSerial2.T + Rpr[0][0])
                    K = P2 @ HprSerial2.T @ S_inv
                    error_states2 = K @ residualprSerial2
                    P2 = (np.eye(13) -  K @ HprSerial2) @ P2
                    P2 = 0.5 * (P2 + P2.T)

                    for k in range(13):
                        x_hat[k] += error_states[k] 
                        x_hat2[ k] += error_states2[k]
                

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

                    dx = x_hat2[0] - estimated_sat_pos[j][0]
                    dy = x_hat2[1] - estimated_sat_pos[j][1]
                    r_nom = np.sqrt(dx**2 + dy**2)
                    ux = dx / r_nom
                    uy = dy / r_nom
                    
                    HdrSerial2[0][2] = ux
                    HdrSerial2[0][3] = uy
                    HdrSerial2[0][12] = 1.0

                    #these sat vels are only valid for a circular flight path
                    satvelX = -GPS.constellation.angularVel * estimated_sat_pos[j][1]  
                    satvelY = GPS.constellation.angularVel * estimated_sat_pos[j][0]
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[12]
                    residualdrSerial = np.array([rawDRs[j] - dr_hat])
                    dr_hat = (ux * (x_hat2[2] - satvelX)) + (uy * (x_hat2[3] - satvelY)) + x_hat2[12]
                    residualdrSerial2 = np.array([rawDRs[j] - dr_hat])
                    
                    # Expected relative velocity
                    # Append the current time and residual to this specific satellite's history
                    history_resdr[j].append(residualdrSerial[0])
                    history_resdr2[j].append(residualdrSerial2[0])
                    
                    # Update the specific line object memory pointers
                    line_objectsdr[j].set_data(history_time[j], history_resdr[j])
                    line_objectsdr2[j].set_data(history_time[j], history_resdr2[j])
                    
                    S_inv = 1.0 / (HdrSerial @ P @ HdrSerial.T + Rdr[0][0])
                    K = P @ HdrSerial.T @ S_inv
                    error_states = K @ residualdrSerial
                    P = (np.eye(13) -  K @ HdrSerial) @ P
                    P = 0.5 * (P + P.T)
                    
                    S_inv = 1.0 / (HdrSerial2 @ P2 @ HdrSerial2.T + Rdr[0][0])
                    K = P2 @ HdrSerial2.T @ S_inv
                    error_states2 = K @ residualdrSerial2
                    P2 = (np.eye(13) -  K @ HdrSerial2) @ P2
                    P2 = 0.5 * (P2 + P2.T)

                    for k in range(13):
                        x_hat[k] += error_states[k] 
                        x_hat2[ k] += error_states2[k]
        

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])
    history_eskf2_x.append(x_hat2[0])
    history_eskf2_y.append(x_hat2[1])

    if i%10==0:
        drone_dot.set_data([curr_x], [curr_y])
        eskf_path.set_data(history_eskf_x, history_eskf_y)
        eskf2_path.set_data(history_eskf2_x, history_eskf2_y)

#    plt.pause(0.0001)
plt.ioff(); plt.show()
