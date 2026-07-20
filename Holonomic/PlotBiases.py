import math
import numpy as np
import matplotlib.pyplot as plt
from IMUv3 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5

#state: x, y, vx, vy, theta, b_ax, b_ay, b_w, b_clk, b_drift
x_hat = np.array([0.0, 0.0, radius * omega, 2 * radius * omega, np.pi/2, 0.0, 0.0, 0.0, 45.0, 0.14])
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 1000.0**2, 100.0**2])

#state: x, y, vx, vy, theta, b_ax, b_ay, S_x, S_y, M_xy b_w, b_clk, b_drift
x_hat2 = np.array([0.0, 0.0, radius * omega, 2 * radius * omega, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0, 0.14])
error_states2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
P2 = np.diag([10.0, 10.0, 1.0, 1.0, 0.1, 1e-4, 1e-4, 1e-5, 4e-4, 4e-4, 4e-4, 1000.0**2, 100.0**2])

dt = 0.01
totalTime = 8 #minutes
IMU = IMUSimulator(dt)
sat_angles = [30, 75, 120, 160]
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
sigma_sf_walk = 0.00001

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

Q2 = np.diag([
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
eskf_path, = ax1.plot([], [], 'm--', label='EKF10state')
eskf2_path, = ax1.plot([], [], 'c--', label='EKF13state')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_eskf2_x, history_eskf2_y = [], []

# Middle Plot: Scale Factor Biases
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-0.05, 0.05)   # Range for scale factors
ax2.set_title("IMU Scale Factor Biases (13 State)")
ax2.set_ylabel("Scale Factor")
ax2.grid(True)

line_Sx_est, = ax2.plot([], [], 'b-', label='Estimated Sx')
line_Sx_true, = ax2.plot([], [], 'b--', label='True Sx')
line_Sy_est, = ax2.plot([], [], 'r-', label='Estimated Sy')
line_Sy_true, = ax2.plot([], [], 'r--', label='True Sy')
ax2.legend(loc='upper right')

# Bottom Plot: Misalignment Bias
ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim(-0.05, 0.05)   # Range for misalignment
ax3.set_title("IMU Misalignment Bias (13 State)")
ax3.set_ylabel("Misalignment")
ax3.grid(True)

line_Mxy_est, = ax3.plot([], [], 'g-', label='Estimated Mxy')
line_Mxy_true, = ax3.plot([], [], 'g--', label='True Mxy')
ax3.legend(loc='upper right')


history_time = []
history_Sx_est, history_Sx_true = [], []
history_Sy_est, history_Sy_true = [], []
history_Mxy_est, history_Mxy_true = [], []

for i in range(int(60 * totalTime / dt)):
    # Extract current state for readability
    t = i * dt
    # 1. Truth
    curr_x = radius * np.sin(omega * t)
    curr_y = radius * np.sin(2 * omega * t)
    curr_theta = np.pi / 2 + t * omega
    curr_velx = radius * omega * np.cos(omega * t)
    curr_vely = 2 * radius * omega * np.cos(2 * omega * t)
    curr_accx = -radius * omega**2 * np.sin(omega * t)
    curr_accy = -radius * (2 * omega)**2 * np.sin(2 * omega * t)
    body_ax = (curr_accx * np.cos(curr_theta)) + (curr_accy * np.sin(curr_theta))
    body_ay = -(curr_accx * np.sin(curr_theta)) + (curr_accy * np.cos(curr_theta))

    ax, ay, omegahat, trueGyroBias, trueAccelXBias, trueAccelYBias, true_Sx, true_Sy, true_Mxy = IMU.generate_measurements(true_ax_body = body_ax, true_ay_body = body_ay, true_omega = omega)

    ax_corr = ax - x_hat[5] # raw_accelx - b_ax
    ay_corr = ay - x_hat[6] # raw_accely - b_ay
    w_corr = omegahat - x_hat[7] # raw_gyro - b_w

    ax_corr2 = ax - ax * x_hat2[8] - ay * x_hat2[10] - x_hat2[5]
    ay_corr2 = ay - ay * x_hat2[9] - ax * x_hat2[10] - x_hat2[6]
    w_corr2 = omegahat - x_hat2[7] # raw_gyro - b_w

    # 2. EKF PREDICT: Move the state forward
    x, y, vx, vy, theta, bax, bay, bw, bclk, drift_clk = x_hat
    x2, y2, vx2, vy2, theta2, bax2, bay2, bw2, Sx, Sy, Mxy, bclk2, drift_clk2 = x_hat2

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

    a_global_x2 = (ax_corr2 * np.cos(theta2)) - (ay_corr2 * np.sin(theta2))
    a_global_y2 = (ax_corr2 * np.sin(theta2)) + (ay_corr2 * np.cos(theta2))
    vx2     += a_global_x2 * dt
    vy2     += a_global_y2 * dt
    theta2 += w_corr2 * dt
    x2     += vx2 * dt
    y2     += vy2 * dt
    bclk2  += drift_clk2 * dt
    x_hat2 = np.array([x2, y2, vx, vy2, theta2, bax2, bay2, bw2, Sx, Sy, Mxy, bclk2, drift_clk2])

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
    P2 = F2 @ P2 @ F2.T + Q2

    active_prns=[0,1,2,3] #includes 0 as a PRN

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0 and i > 0:
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        rawDRs = GPS.get_satellite_DRs(curr_x, curr_y, curr_velx, curr_vely )
        residualpr = np.zeros(len(sat_angles))
        residualdr = np.zeros(len(sat_angles))
        residualpr2 = np.zeros(len(sat_angles))
        residualdr2 = np.zeros(len(sat_angles))

        Hpr = np.zeros((len(sat_angles), 10))
        Hdr = np.zeros((len(sat_angles), 10))
        HprSerial = np.zeros((1,10))
        HdrSerial = np.zeros((1,10))

        Hpr2 = np.zeros((len(sat_angles), 13))
        Hdr2 = np.zeros((len(sat_angles), 13))
        HprSerial2 = np.zeros((1,13))
        HdrSerial2 = np.zeros((1,13))

        sat_x = []
        sat_y = []

        if t < 8 * 60:
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

                    sat_x.append(ux)
                    sat_y.append(uy)

            sat_plot.set_data(sat_x, sat_y)
            history_time.append(t)
            history_Sx_est.append(x_hat2[8])
            history_Sx_true.append(true_Sx)
            history_Sy_est.append(x_hat2[9])
            history_Sy_true.append(true_Sy)
            history_Mxy_est.append(x_hat2[10])
            history_Mxy_true.append(true_Mxy)

            line_Sx_est.set_data(history_time, history_Sx_est)
            line_Sx_true.set_data(history_time, history_Sx_true)
            line_Sy_est.set_data(history_time, history_Sy_est)
            line_Sy_true.set_data(history_time, history_Sy_true)

            line_Mxy_est.set_data(history_time, history_Mxy_est)
            line_Mxy_true.set_data(history_time, history_Mxy_true)
            K = P @ Hpr.T @ np.linalg.inv (Hpr @ P @ Hpr.T + Rpr)
            error_states = K @ residualpr.T
            P = (np.eye(10) -  K @ Hpr) @ P
            P = 0.5 * (P + P.T)

            K = P2 @ Hpr2.T @ np.linalg.inv (Hpr2 @ P2 @ Hpr2.T + Rpr)
            error_states2 = K @ residualpr2.T
            P2 = (np.eye(13) -  K @ Hpr2) @ P2
            P2 = 0.5 * (P2 + P2.T)

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

            x_hat2[0] += error_states2[0]  # Fix X
            x_hat2[ 1] += error_states2[1]  # Fix Y
            x_hat2[ 2] += error_states2[2]  # Fix Velocity
            x_hat2[ 3] += error_states2[3]  # Fix Heading
            x_hat2[ 4] += error_states2[4]
            x_hat2[ 5] += error_states2[5]
            x_hat2[ 6] += error_states2[6]
            x_hat2[ 7] += error_states2[7]
            x_hat2[ 8] += error_states2[8]
            x_hat2[ 9] += error_states2[9]
            x_hat2[ 10] += error_states2[10]
            x_hat2[ 11] += error_states2[11]
            x_hat2[ 12] += error_states2[12]

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
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[9]
                    dr_hat2 = (ux * (x_hat2[2] - satvelX)) + (uy * (x_hat2[3] - satvelY)) + x_hat2[12]
                    residualdr[j] = rawDRs[j] - dr_hat
                    residualdr2[j] = rawDRs[j] - dr_hat2


            K = P @ Hdr.T @ np.linalg.inv (Hdr @ P @ Hdr.T + Rdr)
            error_states = K @ residualdr.T
            P = (np.eye(10) -  K @ Hdr) @ P
            P = 0.5 * (P + P.T)

            K = P2 @ Hdr2.T @ np.linalg.inv (Hdr2 @ P2 @ Hdr2.T + Rdr)
            error_states2 = K @ residualdr2.T
            P2 = (np.eye(13) -  K @ Hdr2) @ P2
            P2 = 0.5 * (P2 + P2.T)

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

            x_hat2[0] += error_states2[0]  # Fix X
            x_hat2[ 1] += error_states2[1]  # Fix Y
            x_hat2[ 2] += error_states2[2]  # Fix Velocity
            x_hat2[ 3] += error_states2[3]  # Fix Heading
            x_hat2[ 4] += error_states2[4]
            x_hat2[ 5] += error_states2[5]
            x_hat2[ 6] += error_states2[6]
            x_hat2[ 7] += error_states2[7]
            x_hat2[ 8] += error_states2[8]
            x_hat2[ 9] += error_states2[9]
            x_hat2[ 10] += error_states2[10]
            x_hat2[ 11] += error_states2[11]
            x_hat2[ 12] += error_states2[12]

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

                    S_inv = 1.0 / (HprSerial @ P @ HprSerial.T + Rpr[0][0])
                    K = P @ HprSerial.T @ S_inv
                    error_states = K @ residualprSerial
                    P = (np.eye(10) -  K @ HprSerial) @ P
                    P = 0.5 * (P + P.T)

                    S_inv = 1.0 / (HprSerial2 @ P2 @ HprSerial2.T + Rpr[0][0])
                    K = P2 @ HprSerial2.T @ S_inv
                    error_states2 = K @ residualprSerial2
                    P2 = (np.eye(13) -  K @ HprSerial2) @ P2
                    P2 = 0.5 * (P2 + P2.T)

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

                    x_hat2[0] += error_states2[0]  # Fix X
                    x_hat2[ 1] += error_states2[1]  # Fix Y
                    x_hat2[ 2] += error_states2[2]  # Fix Velocity
                    x_hat2[ 3] += error_states2[3]  # Fix Heading
                    x_hat2[ 4] += error_states2[4]
                    x_hat2[ 5] += error_states2[5]
                    x_hat2[ 6] += error_states2[6]
                    x_hat2[ 7] += error_states2[7]
                    x_hat2[ 8] += error_states2[8]
                    x_hat2[ 9] += error_states2[9]
                    x_hat2[ 10] += error_states2[10]
                    x_hat2[ 11] += error_states2[11]
                    x_hat2[ 12] += error_states2[12]


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
                    dr_hat = (ux * (x_hat[2] - satvelX)) + (uy * (x_hat[3] - satvelY)) + x_hat[9]
                    residualdrSerial = np.array([rawDRs[j] - dr_hat])
                    dr_hat = (ux * (x_hat2[2] - satvelX)) + (uy * (x_hat2[3] - satvelY)) + x_hat2[12]
                    residualdrSerial2 = np.array([rawDRs[j] - dr_hat])

                    S_inv = 1.0 / (HdrSerial @ P @ HdrSerial.T + Rdr[0][0])
                    K = P @ HdrSerial.T @ S_inv
                    error_states = K @ residualdrSerial
                    P = (np.eye(10) -  K @ HdrSerial) @ P
                    P = 0.5 * (P + P.T)

                    S_inv = 1.0 / (HdrSerial2 @ P2 @ HdrSerial2.T + Rdr[0][0])
                    K = P2 @ HdrSerial2.T @ S_inv
                    error_states2 = K @ residualdrSerial2
                    P2 = (np.eye(13) -  K @ HdrSerial2) @ P2
                    P2 = 0.5 * (P2 + P2.T)

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

                    x_hat2[0] += error_states2[0]  # Fix X
                    x_hat2[ 1] += error_states2[1]  # Fix Y
                    x_hat2[ 2] += error_states2[2]  # Fix Velocity
                    x_hat2[ 3] += error_states2[3]  # Fix Heading
                    x_hat2[ 4] += error_states2[4]
                    x_hat2[ 5] += error_states2[5]
                    x_hat2[ 6] += error_states2[6]
                    x_hat2[ 7] += error_states2[7]
                    x_hat2[ 8] += error_states2[8]
                    x_hat2[ 9] += error_states2[9]
                    x_hat2[ 10] += error_states2[10]
                    x_hat2[ 11] += error_states2[11]
                    x_hat2[ 12] += error_states2[12]


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
