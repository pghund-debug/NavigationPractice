import numpy as np

# Simulate 4 GPS satellites orbiting at fixed angles far away
# The center of the earth/grid is roughly (0,0) and the orbit radius is massive
GPS_ORBIT_RADIUS = 20_200_000  # True GPS altitude is ~20,200 km


class satConstellation:
    def __init__(self, dt, angles):
        # Fixed angular positions of satellites in the sky
        self.sat_angles = []
        for j in range(len(angles)):
            self.sat_angles.append(np.radians(angles[j]))

        #compute HDOP
        H = np.zeros((len(angles), 3))
        if len(self.sat_angles) < 3:
            print("HDOP is infinite")
        else:
            for i, angle in enumerate(self.sat_angles):
                H[i, 0] = np.cos(angle)  # X unit vector
                H[i, 1] = np.sin(angle)  # Y unit vector
                H[i, 2] = 1.0            # Clock multiplier

            try:
                # 2. Compute the Cofactor Matrix: Q = (H^T * H)^-1
                # H.T @ H results in a 3x3 matrix
                Q = np.linalg.inv(H.T @ H)
                
                # 3. Extract diagonal elements and calculate HDOP
                hdop = np.sqrt(Q[0, 0] + Q[1, 1])
                print("HDOP is %.4f" % hdop)
                
            except np.linalg.LinAlgError:
                # If the satellites are perfectly aligned (e.g. all on exactly one side),
                # the matrix becomes singular/uninvertible. Geometry is infinitely bad.
                print("HDOP is infinite")
        self.dt = dt
        self.angularVel = 0.01

    def get_satellite_positions(self):
        sat_positions = []
        for i in range(len(self.sat_angles)):
            # To simulate realistic movement, we can let the satellites drift slowly over time
            self.sat_angles[i] += self.angularVel 
            
            sat_x = GPS_ORBIT_RADIUS * np.cos(self.sat_angles[i])
            sat_y = GPS_ORBIT_RADIUS * np.sin(self.sat_angles[i])
            
            sat_positions.append([sat_x, sat_y])
            
        return np.array(sat_positions)


class GPSR:
    def __init__(self, dt, angles):
        self.dt = dt
        self.true_clock_bias = 45.0
        self.true_clock_walk_drift = 0.15
        self.constellation = satConstellation(dt, angles)
        self.timeFactor = np.sqrt(dt)
        self.numSats = len(angles)
        # Generate a static Ephemeris Error for each satellite (e.g., ~2 meters of error)
        self.ephemeris_errors = np.random.normal(0, 2.0, (self.numSats, 2)) # [dx, dy] for each sat

    def get_satellite_positions(self, x_true, y_true):
        sat_positions = self.constellation.get_satellite_positions()
        self.true_clock_walk_drift += np.random.normal(0, 0.05) * self.timeFactor
        self.true_clock_bias += (self.true_clock_walk_drift * self.dt) + np.random.normal(0, 0.1) * self.timeFactor
        raw_prs = np.zeros(self.numSats)
        estimated_sat_pos = np.zeros((self.numSats, 2), dtype=np.float64)

        for i in range(len(sat_positions)):
            # True geometric range from the actual drone to the satellite
            r_true = np.sqrt((x_true - sat_positions[i][0])**2 + (y_true - sat_positions[i][1])**2)
            
            # The hardware measurement is corrupted by the TRUE clock bias and antenna white noise!
            antenna_noise = np.random.normal(0, 0.5)
            raw_prs[i] = r_true + self.true_clock_bias + antenna_noise
            estimated_sat_pos[i][0] = sat_positions[i][0] - self.ephemeris_errors[i][0]
            estimated_sat_pos[i][1] = sat_positions[i][1] - self.ephemeris_errors[i][1]
        
        return raw_prs, estimated_sat_pos, self.true_clock_bias


    def get_satellite_DRs(self, x_true, y_true, velx_true, vely_true):
        
        raw_drs = np.zeros(self.numSats)
        sat_positions = self.constellation.get_satellite_positions()

        for i in range(self.numSats):

            dx = x_true - sat_positions[i][0]
            dy = y_true - sat_positions[i][1]
            r_true = np.sqrt(dx**2 + dy**2)
            ux_true = dx / r_true
            uy_true = dy / r_true
            
            sat_vx = -self.constellation.angularVel * GPS_ORBIT_RADIUS * np.sin(self.constellation.sat_angles[i]) 
            sat_vy =  self.constellation.angularVel * GPS_ORBIT_RADIUS * np.cos(self.constellation.sat_angles[i])
            
            v_rel_x = velx_true - sat_vx
            v_rel_y = vely_true - sat_vy

            true_doppler = (v_rel_x * ux_true) + (v_rel_y * uy_true)

            antenna_noise = np.random.normal(0, 0.05)
            raw_drs[i] = true_doppler + self.true_clock_walk_drift + antenna_noise 

        return raw_drs
