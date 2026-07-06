import numpy as np

# Simulate 4 GPS satellites orbiting at fixed angles far away
# The center of the earth/grid is roughly (0,0) and the orbit radius is massive
GPS_ORBIT_RADIUS = 20_200_000  # True GPS altitude is ~20,200 km


class satConstellation:
    def __init__(self, dt):
        # Fixed angular positions of 4 satellites in the sky
        self.sat_angles = [np.radians(30), np.radians(75), np.radians(120), np.radians(160)]
        self.dt = dt

    def get_satellite_positions():
        sat_positions = []
        for i in range(len(sat_angles)):
            # To simulate realistic movement, we can let the satellites drift slowly over time
            self.sat_angles[i] += 0.0001 
            
            sat_x = GPS_ORBIT_RADIUS * np.cos(self.sat_angles[i])
            sat_y = GPS_ORBIT_RADIUS * np.sin(self.sat_angles[i])
            
            sat_positions.append([sat_x, sat_y])
            
        return np.array(sat_positions)


class GPSR:
    def __init__(self, dt):
        self.dt = dt
        self.true_clock_bias = 45.0
        self.true_clock_walk_drift = 0.15
        self.constellation = satConstellation(dt)
        self.timeFactor = np.sqrt(dt)
        self.numSats = 4
        # Generate a static Ephemeris Error for each satellite (e.g., ~2 meters of error)
        ephemeris_errors = np.random.normal(0, 2.0, (self.numSats, 2)) # [dx, dy] for each sat

    def get_satellite_positions(x_true, y_true):
        sat_positions = self.constellation.get_satellite_positions()
        self.true_clock_bias += np.random.normal(0, self.true_clock_walk_drift) * self.timeFactor
        raw_prs = np.zeros(self.numSats)
        estimated_sat_pos = np.zeros((self.numSats, 2), dtype=np.float64)

        for i in range(len(sat_positions)):
            # True geometric range from the actual drone to the satellite
            r_true = np.sqrt((x_true - sat_positions[i][0])**2 + (y_true - sat_positions[i][1])**2)
            
            # The hardware measurement is corrupted by the TRUE clock bias and antenna white noise!
            antenna_noise = np.random.normal(0, 0.5)
            raw_prs[i] = r_true + true_clock_bias + antenna_noise
            estimate_sat_pos[i][0] = sat_positions[i][0] - ephemeris_errors[i][0]
            estimate_sat_pos[i][1] = sat_positions[i][1] - ephemeris_errors[i][1]
        
        return raw_prs, estimated_sat_pos, true_clock_bias
