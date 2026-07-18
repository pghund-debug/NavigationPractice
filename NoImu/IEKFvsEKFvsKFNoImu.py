import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

radius = 20
omega = 0.5

# --- KF Initialization ---
# State x = [pos_x, pos_y, vel_x, vel_y]
x_hat = np.array([[radius], [0.0], [0.0], [radius * omega]]) 
P = np.eye(4) * 1.0  # Initial uncertainty

# --- EKF Initialization ---
# State: [x, y, velocity, heading]
xe_hat = np.array([[radius], [0.0], [radius * 0.5], [np.pi/2]])
PE = np.eye(4) * 0.1

# --- IEKF State Initialization ---
# R = Identity (facing North), v = [0,0, 0], p = [radius,0, 0]
X_hat = np.eye(5)
X_hat[:2, :2] = [[0, -1], [1, 0]] # Initial 90 deg rotation
X_hat[:3, 4] = [radius, 0, 0]           # Initial Position
X_hat[:3, 3] = [radius * 0.5, 0.0, 0.0]
# Covariance (now 'Lie Algebra' error)
PI = np.eye(9) * 0.1

history_kf_x, history_kf_y = [], []
history_ekf_x, history_ekf_y = [], []
history_iekf_x, history_iekf_y = [], []
res_history_x, res_history_y, res_t = [], [], []
rese_history_x, rese_history_y = [], []
resi_history_x, resi_history_y = [], []

def iekf_calculate_correction(X, P, z, H, R):
    """
    X: Current 5x5 State Matrix [R v p; 0 1 0; 0 0 1]
    P: 9x9 Covariance Matrix
    z: 3x1 GPS measurement vector [x, y, z]
    H: 3x9 Constant Measurement Matrix
    R: 3x3 Measurement Noise Covariance
    """
    # 1. Extract components from the 5x5 matrix
    R_curr = X[0:3, 0:3]
    v_curr = X[0:3, 3]
    p_curr = X[0:3, 4]

    # 2. Compute the INVARIANT INNOVATION (The "Inverse" trick)
    # Instead of (z - p), we rotate the error into the body frame:
    # y = R.T @ (z - p)
    # This makes the innovation independent of the global heading.
    innovation = R_curr.T @ (z.flatten() - p_curr)

    resi_history_x.append(innovation[0])
    resi_history_y.append(innovation[1])
    
    # 3. Standard Kalman Gain calculation
    # Because H is constant [0 0 I], this is very efficient.
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # 4. Calculate the 9D error vector (Lie Algebra)
    delta = K @ innovation

    return delta, K


def skew_symmetric(v):
    # Converts a 3D vector [x, y, z] into its 3x3 skew-symmetric matrix.
    # This represents the angular velocity cross-product matrix.
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def exp_map_so3(w):
    # Rodrigo's Formula for the SO(3) exponential map.
    # Maps an angular error vector (w) to a 3x3 Rotation Matrix (R).
    theta = norm(w)
    if theta < 1e-6: # Taylor expansion for near-zero rotation
        return np.eye(3) + skew_symmetric(w)

    w_skew = skew_symmetric(w)
    # The geometric logic of the curve
    R = np.eye(3) + (np.sin(theta)/theta) * w_skew + ((1-np.cos(theta))/(theta**2)) * (w_skew @ w_skew)
    return R

def exp_map_se2_3(delta):
    # The Full Exponential Map for SE2(3).
    # delta is the 9D vector: [xi_theta (3D), xi_v (3D), xi_p (3D)]
    # Returns a 5x5 Transformation Matrix.
    # 1. Unpack the vector
    xi_theta = delta[0:3]
    xi_v = delta[3:6]
    xi_p = delta[6:9]

    # 2. Get the 3x3 Rotation Matrix (Geometric Truth)
    R = exp_map_so3(xi_theta)

    # 3. Calculate the translational 'shifters'
    # These map the non-linear coupling between rotation, velocity, and position.
    theta = norm(xi_theta)
    if theta < 1e-6:
        # Near-linear approximation
        V = np.eye(3) + skew_symmetric(xi_theta) / 2.0
    else:
        # Full geometric coupling
        w_skew = skew_symmetric(xi_theta)
        w_skew2 = w_skew @ w_skew

        V = np.eye(3) + ((1-np.cos(theta))/(theta**2)) * w_skew + ((theta-np.sin(theta))/(theta**3)) * w_skew2

    # 4. Update Translational Parts
    # Velocity and Position errors are "shunted" by the rotation error.
    v_new = V @ xi_v
    p_new = V @ xi_p

    # 5. Build the 5x5 Matrix (The final IEKF state block)
    # It looks like [ R v p; 0 1 0; 0 0 1 ]
    M = np.eye(5)
    M[0:3, 0:3] = R      # Orientation
    M[0:3, 3] = v_new    # Velocity
    M[0:3, 4] = p_new    # Position

    return M

# State Transition Matrix (Constant Velocity Model)
dt = 0.05
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Measurement Matrix (We only measure position via GPS)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

HI = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Noise Covariances
Q = np.eye(4) * 0.1  # Process noise (how much we trust our physics)
QE = np.diag([0.01, 0.01, 0.1, 0.05]) # Uncertainty in physics
QI = np.eye(9) * 0.1  # Process noise (how much we trust our physics)
R = np.eye(2) * 2.25  # Measurement noise (GPS variance: 1.5^2)
RI = np.eye(3) * 2.25  # Measurement noise (GPS variance: 1.5^2)


# --- Real-Time Loop ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
fig.tight_layout(pad=4.0)

# Plot handles
# Top Plot: Navigation
ax1.set_xlim(-radius * 1.5, radius * 1.5); ax1.set_ylim(-radius * 1.5, radius * 1.5)
ax1.set_title("Drone Navigation")
ax1.grid(True)
kf_path, = ax1.plot([], [], 'b--', label='Simple KF')
ekf_path, = ax1.plot([], [], 'k--', label='EKF')
iekf_path, = ax1.plot([], [], 'c--', label='IEKF')
drone_dot, = ax1.plot([], [], 'go', label='Truth')
gps_dot, = ax1.plot([], [], 'rx', alpha=0.5, label='GPS')
ax1.legend()


# Bottom Plot: Residuals (z - Hx)
ax2.set_xlim(0, 300) # Number of simulation steps
ax2.set_ylim(-radius * 0.5, radius * 0.5)   # Error in meters
ax2.set_title("GPS Residuals (Innovation)")
ax2.set_ylabel("Error (m)")
res_x_line, = ax2.plot([], [], 'b-', label='KFX-Residual', alpha=0.6)
#res_y_line, = ax2.plot([], [], 'b-', label='KFY-Residual', alpha=0.6)
rese_x_line, = ax2.plot([], [], 'k-', label='EKFX-Residual', alpha=0.6)
#rese_y_line, = ax2.plot([], [], 'k-', label='EKFY-Residual', alpha=0.6)
resi_x_line, = ax2.plot([], [], 'c-', label='IEKFX-Residual', alpha=0.6)
#resi_y_line, = ax2.plot([], [], 'c-', label='IEKFY-Residual', alpha=0.6)
ax2.legend()

I4 = np.eye(4)
A = np.zeros((9, 9))
for i in range(300):
    # Extract current state for readability
    _, _, v, theta = xe_hat.flatten()
    t = i * dt
    # 1. Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    
    # 2. KF PREDICT
    x_hat = F @ x_hat
    P = F @ P @ F.T + Q

    # EKF PREDICT: Move the state forward using trig
    xe_hat[0,0] += v * np.cos(theta) * dt
    xe_hat[1,0] += v * np.sin(theta) * dt
    # xe_hat[2,0] (velocity) and xe_hat[3,0] (theta) stay same in constant model
   
    # IEKF PREDICT: Using Group Multiplication
    # Instead of adding, we "multiply" the motion onto the state
    # Omega is our rotation rate

    # Update Rotation
    #Omega_skew = skew_symmetric([0, 0, omega]) # assuming z-axis rotation
    R_old = X_hat[:3, :3].copy()
    X_hat[:3, :3] = R_old @ exp_map_so3([0,0,omega * dt])
    # Update Position (p = p + R*v*dt)
    X_hat[:3, 4] += R_old @ X_hat[:3, 3] * dt
    # The A matrix here is CONSTANT for many navigation models!
    A.fill(0)
    # Fill A with the "physics" of the Lie Algebra
    PI = PI + (A @ PI + PI @ A.T + QI) * dt

    # 2. LINEARIZE (The Jacobian G)
    # This is the derivative of the physics above
    G = np.array([
        [1, 0, np.cos(theta)*dt, -v*np.sin(theta)*dt],
        [0, 1, np.sin(theta)*dt,  v*np.cos(theta)*dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Update Covariance using the Jacobian
    PE = G @ PE @ G.T + QE

    # 3. UPDATE (Every 10 frames when GPS "arrives")
    if i % 10 == 0:
        z = np.array([[curr_x + np.random.normal(0, 1.5)], 
                      [curr_y + np.random.normal(0, 1.5)]])

        z_p = np.array([z[0], z[1], [0]]) # 3D GPS measurement 
        # RESIDUAL calculation (This is the "Surprise")
        residual = z - H @ x_hat
        residuale = z - H @ xe_hat

        # Gain and Update
        #KF
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_hat = x_hat + K @ (z - H @ x_hat)
        P = (I4 - K @ H) @ P
        
        #EKF
        SE = H @ PE @ H.T + R
        KE = PE @ H.T @ np.linalg.inv(SE)
        xe_hat = xe_hat + KE @ (z - H @ xe_hat)
        PE = (I4 - KE @ H) @ PE
        
        #IEKF
        #returns the error vector 'delta' calculated by Kalman Gain K
        delta, KI = iekf_calculate_correction(X_hat, PI, z_p, HI, RI)
        # delta is a 9D vector. M is a 5x5 group matrix.
        M_correction = exp_map_se2_3(delta)
        # We multiply the matrix onto the state, rather than adding vectors.
        # This keeps the geometry perfect.
        X_hat = X_hat @ M_correction

        # Store residuals for plotting
        res_t.append(i)
        res_history_x.append(residual[0,0])
        res_history_y.append(residual[1,0])
        rese_history_x.append(residuale[0,0])
        rese_history_y.append(residuale[1,0])

        gps_dot.set_data([z[0,0]], [z[1,0]])

    # 4. Visualization
    history_kf_x.append(x_hat[0,0])
    history_kf_y.append(x_hat[1,0])
    history_ekf_x.append(xe_hat[0,0])
    history_ekf_y.append(xe_hat[1,0])
    history_iekf_x.append(X_hat[0,4])
    history_iekf_y.append(X_hat[1,4])
    
    kf_path.set_data(history_kf_x, history_kf_y)
    ekf_path.set_data(history_ekf_x, history_ekf_y)
    iekf_path.set_data(history_iekf_x, history_iekf_y)
    drone_dot.set_data([curr_x], [curr_y])

    if res_t: # Only plot if we have data
        res_x_line.set_data(res_t, res_history_x)
        #res_y_line.set_data(res_t, res_history_y)
        rese_x_line.set_data(res_t, rese_history_x)
        #rese_y_line.set_data(res_t, rese_history_y)
        resi_x_line.set_data(res_t, resi_history_x)
        #resi_y_line.set_data(res_t, resi_history_y)
    
    plt.pause(0.03)

plt.ioff(); plt.show()
