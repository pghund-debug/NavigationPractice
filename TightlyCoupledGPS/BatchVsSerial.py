import numpy as np
import matplotlib.pyplot as plt
import time

from IMUv1 import IMUSimulator
from GPSv1 import GPSR

radius = 20
omega = 0.5
plotBiases = False

# --- EKF Initialization ---
x_hat = np.array([radius, 0.0, radius * omega, np.pi/2, 0.0, 0.0, 40.0])  
error_states = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  #[dx  dy dv dtheta dba dbw dbclck]
Perror = np.diag([10.0, 10.0, 1.0, 0.1, 1e-4, 1e-5, 1000.0**2])

dt = 0.01
totalTime = 1 #minutes
IMU = IMUSimulator(dt)
GPS = GPSR(dt * 100, [30,  75, 120, 160 ])

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
Rserial = 2.0**2  # Measurement noise
R = np.diag([4.0, 4.0, 4.0, 4.0])

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
sat1_dot, = ax1.plot([], [], 'ro', label='Sat1')
sat2_dot, = ax1.plot([], [], 'co', label='Sat2')
sat3_dot, = ax1.plot([], [], 'mo', label='Sat3')
sat4_dot, = ax1.plot([], [], 'yo', label='Sat4')
eskf_path, = ax1.plot([], [], 'b--', label='ESKF')
ax1.legend()

history_eskf_x, history_eskf_y = [], []
history_ecb, history_cb = [], []
history_res1, history_res2, history_res3, history_res4, res_t = [],[],[], [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals (Innovation)")
ax2.set_ylabel("Error (m)")
res1_line, = ax2.plot([], [], 'r-', label='Sat1-Residual', alpha=0.6)
res2_line, = ax2.plot([], [], 'c-', label='Sat2-Residual', alpha=0.6)
res3_line, = ax2.plot([], [], 'm-', label='Sat3-Residual', alpha=0.6)
res4_line, = ax2.plot([], [], 'y-', label='Sat4-Residual', alpha=0.6)
ax2.legend()

ax3.set_xlim(0, 60 * totalTime) # Number of simulation steps
ax3.set_ylim( 40, 50)   # Error in meters
ax3.set_title("Clock Bias")
ax3.set_ylabel("Error (m)")
ecb_line, = ax3.plot([], [], 'b-', label='Estimated Clock Bias', alpha=0.6)
cb_line, = ax3.plot([], [], 'r-', label='Clock Bias', alpha=0.6)
ax3.legend()

I7 = np.eye(7)
residual = np.zeros(4)
H = np.zeros((4, 7))
Hserial = np.zeros(7)
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
    x, y, v, theta, ba, bw, bclk = x_hat
    
    x     += v * np.cos(theta) * dt
    y     += v * np.sin(theta) * dt
    v     += a_corr * dt
    theta += w_corr * dt
    x_hat = np.array([x, y, v, theta, ba, bw, bclk])

    # 2. LINEARIZE
    # This is the derivative of the physics above
    F = I7.copy()
    F[0, 2] = np.cos(theta) * dt
    F[0, 3] = -v * np.sin(theta) * dt
    F[1, 2] = np.sin(theta) * dt
    F[1, 3] = v * np.cos(theta) * dt
    F[2, 4] = -dt
    F[3, 5] = -dt

    # Update Covariance using the Jacobian
    Perror = F @ Perror @ F.T + Q

    # 3. KF UPDATE (Every 100 frames when GPS "arrives")
    if i % int(1/dt) == 0:
        print("GPS update") 
        rawPRs, estimated_sat_pos, true_clock_bias = GPS.get_satellite_positions(curr_x, curr_y)
        residual.fill(0)
        H.fill(0)
        Hserial.fill(0)
        if t < 5: #for the first five seconds, use batch incorporation for stability
            #batch incorporation of measurements
            start = time.perf_counter()
            for j in range(len(estimated_sat_pos)):
                dx = x_hat[0] - estimated_sat_pos[j][0]
                dy = x_hat[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                H[j][0] = ux
                H[j][1] = uy
                H[j][6] = 1

                pr_hat = r_nom + x_hat[6]
                ux*=30
                uy*=30
                residual[j] = rawPRs[j] - pr_hat
                if j==0:    
                    sat1_dot.set_data([ux], [uy])
                    history_res1.append(residual[j])
                elif j==1:    
                    sat2_dot.set_data([ux], [uy])
                    history_res2.append(residual[j])
                elif j==2:    
                    sat3_dot.set_data([ux], [uy])
                    history_res3.append(residual[j])
                else:    
                    sat4_dot.set_data([ux], [uy])
                    history_res4.append(residual[j])

           
            K = Perror @ H.T @ np.linalg.inv (H @ Perror @ H.T + R)
            error_states = K @ residual.T
            Perror = (I7 -  K @ H) @ Perror
            # --- INJECTION STEP ---
            # Apply error estimations directly to nominal totals
            x_hat[0] += error_states[0]  # Fix X
            x_hat[ 1] += error_states[1]  # Fix Y
            x_hat[ 2] += error_states[2]  # Fix Velocity
            x_hat[ 3] += error_states[3]  # Fix Heading
            x_hat[ 4] += error_states[4]
            x_hat[ 5] += error_states[5]  
            x_hat[ 6] += error_states[6]

            print("batch processing time: %.6f seconds" % float(time.perf_counter() - start))

            history_ecb.append(x_hat[6])
            history_cb.append(true_clock_bias)
        
        else: #switch to serial incorporation
            #serial incorporation of measurements
            start = time.perf_counter()
            for j in range(len(estimated_sat_pos)):
                dx = x_hat[0] - estimated_sat_pos[j][0]
                dy = x_hat[1] - estimated_sat_pos[j][1]
                r_nom = np.sqrt(dx**2 + dy**2)
                ux = dx / r_nom
                uy = dy / r_nom
                Hserial[0] = ux
                Hserial[1] = uy
                Hserial[6] = 1

                pr_hat = r_nom + x_hat[6]
                residual = np.sqrt((curr_x - estimated_sat_pos[j][0])**2 + (curr_y - estimated_sat_pos[j][1])**2) - pr_hat
                Kserial = Perror @ Hserial.T / (Hserial @ Perror @ Hserial.T + Rserial)
                error_states = Kserial * residual
                Perror = (I7 -  Kserial @ Hserial) @ Perror
                # --- INJECTION STEP ---
                # Apply error estimations directly to nominal totals
                x_hat[0] += error_states[0]  # Fix X
                x_hat[ 1] += error_states[1]  # Fix Y
                x_hat[ 2] += error_states[2]  # Fix Velocity
                x_hat[ 3] += error_states[3]  # Fix Heading
                x_hat[ 4] += error_states[4]
                x_hat[ 5] += error_states[5]  
                x_hat[ 6] += error_states[6]
               
                if j==0:    
                    sat1_dot.set_data([ux], [uy])
                    history_res1.append(residual)
                elif j==1:    
                    sat2_dot.set_data([ux], [uy])
                    history_res2.append(residual)
                elif j==2:    
                    sat3_dot.set_data([ux], [uy])
                    history_res3.append(residual)
                else:    
                    sat4_dot.set_data([ux], [uy])
                    history_res4.append(residual)
            print("serial processing time: %.6f seconds" % float(time.perf_counter() - start) )

        # Store residuals for plotting
        res_t.append(t)
        Perror = 0.5 * (Perror + Perror.T)

    # 4. Visualization
    history_eskf_x.append(x_hat[0])
    history_eskf_y.append(x_hat[1])

    drone_dot.set_data([curr_x], [curr_y])
    eskf_path.set_data(history_eskf_x, history_eskf_y)
    if res_t:
        res1_line.set_data(res_t, history_res1)
        res2_line.set_data(res_t, history_res2)
        res3_line.set_data(res_t, history_res3)
        res4_line.set_data(res_t, history_res4)
        ecb_line.set_data(res_t, history_ecb)
        cb_line.set_data(res_t, history_cb)

    plt.pause(0.0001)
plt.ioff(); plt.show()
