import numpy as np
import matplotlib.pyplot as plt

radius = 20
omega = 0.5

# --- KF Initialization ---
# State x = [pos_x, pos_y, vel_x, vel_y]
x_hat = np.array([[radius * omega], [0.0], [0.0], [0.0]]) 
P = np.eye(4) * 1.0  # Initial uncertainty
x2_hat = np.array([[radius * omega], [0.0], [0.0], [0.0]]) 
P2 = np.eye(4) * 1.0  # Initial uncertainty

# State Transition Matrix (Constant Velocity Model)
dt = 0.05
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Measurement Matrix (We only measure position via GPS)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Noise Covariances
Q = np.eye(4) * 2.25  # Process noise (how much we trust our physics)
Q2 = np.eye(4) * 0.05  # Process noise (how much we trust our physics)
R = np.eye(2) * 0.01  # Measurement noise (GPS variance: 1.5^2)
R2 = np.eye(2) * 2.25  # Measurement noise (GPS variance: 1.5^2)

# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
kf_path, = ax1.plot([], [], 'b--', label='Simple KF 1')
kf2_path, = ax1.plot([], [], 'k--', label='Simple KF 2')
drone_dot, = ax1.plot([], [], 'go', label='Truth')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()

history_kf_x, history_kf_y = [], []
history_kf2_x, history_kf2_y = [], []
res_history_x, res_history_y, res_t = [], [], []
res2_history_x, res2_history_y, res2_t = [], [], []

# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 300) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals (Innovation)")
ax2.set_ylabel("Error (m)")
res_x_line, = ax2.plot([], [], 'b-', label='X-Residual', alpha=0.6)
#res_y_line, = ax2.plot([], [], 'b-', label='Y-Residual', alpha=0.6)
res2_x_line, = ax2.plot([], [], 'k-', label='X2-Residual', alpha=0.6)
#res2_y_line, = ax2.plot([], [], 'k-', label='Y2-Residual', alpha=0.6)
ax2.legend()

I4 = np.eye(4)
for i in range(300):
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    # 2. KF PREDICT
    x_hat = F @ x_hat
    x2_hat = F @ x2_hat
    P = F @ P @ F.T + Q
    P2 = F @ P2 @ F.T + Q2
    
    # 3. KF UPDATE (Every 10 frames when GPS "arrives")
    if i % 10 == 0:
        z = np.array([[curr_x + np.random.normal(0, 0.1)], 
                      [curr_y + np.random.normal(0, 0.1)]])
       
        # RESIDUAL calculation (This is the "Surprise")
        residual = z - H @ x_hat
        residual2 = z - H @ x2_hat

        # Gain and Update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_hat = x_hat + K @ (z - H @ x_hat)
        P = (I4 - K @ H) @ P
        
        S2 = H @ P2 @ H.T + R2
        K2 = P2 @ H.T @ np.linalg.inv(S2)
        x2_hat = x2_hat + K2 @ (z - H @ x2_hat)
        P2 = (I4 - K2 @ H) @ P2
       
        # Store residuals for plotting
        res_t.append(i)
        res_history_x.append(residual[0,0])
        res_history_y.append(residual[1,0])
        res2_t.append(i)
        res2_history_x.append(residual2[0,0])
        res2_history_y.append(residual2[1,0])

        gps_dot.set_data([z[0,0]], [z[1,0]])

    # 4. Visualization
    history_kf_x.append(x_hat[0,0])
    history_kf_y.append(x_hat[1,0])
    history_kf2_x.append(x2_hat[0,0])
    history_kf2_y.append(x2_hat[1,0])
    
    kf_path.set_data(history_kf_x, history_kf_y)
    kf2_path.set_data(history_kf2_x, history_kf2_y)
    drone_dot.set_data([curr_x], [curr_y])

    if res_t: # Only plot if we have data
        res_x_line.set_data(res_t, res_history_x)
        #res_y_line.set_data(res_t, res_history_y)
        res2_x_line.set_data(res2_t, res2_history_x)
        #res2_y_line.set_data(res2_t, res2_history_y)
    
    plt.pause(0.03)

plt.ioff(); plt.show()
