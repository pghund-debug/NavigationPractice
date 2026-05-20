import numpy as np
import matplotlib.pyplot as plt

# Setup the figure
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.grid(True)

# Initialize plot elements
path_line, = ax.plot([], [], 'k-', label='True Path', alpha=0.3)
drone_dot, = ax.plot([], [], 'go', markersize=10, label='Drone')
gps_dot,   = ax.plot([], [], 'rx', label='GPS Ping')

# Storage for the "trail"
history_x, history_y = [], []

# Simulation Parameters
dt = 0.05 
radius = 10.0
omega = 0.5

for i in range(300):
    t = i * dt
    
    # 1. Update Ground Truth
    curr_x = radius * np.cos(omega * t)
    curr_y = radius * np.sin(omega * t)
    history_x.append(curr_x)
    history_y.append(curr_y)
    
    # 2. Update Plot Data
    path_line.set_data(history_x, history_y)
    drone_dot.set_data([curr_x], [curr_y])
    
    # 3. Simulate low-freq GPS (every 10 frames)
    if i % 10 == 0:
        noise_x = curr_x + np.random.normal(0, 1.5)
        noise_y = curr_y + np.random.normal(0, 1.5)
        gps_dot.set_data([noise_x], [noise_y])
    
    # 4. The "Real-Time" Secret Sauce
    plt.pause(0.01) # Briefly pause to allow the GUI to redraw

plt.ioff() # Turn off interactive mode when done
plt.show()
