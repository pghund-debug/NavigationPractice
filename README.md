# Drone Navigation Simulator

This project is a collection of Python scripts simulating drone and vehicle navigation using various Kalman Filter implementations. It demonstrates the progression from standard Kalman Filters to Tightly Coupled GPS configurations.

## Directory Structure

*   `NoImu/`: Simulations of Kalman Filter (KF) and Extended Kalman Filter (EKF) without an Inertial Measurement Unit (IMU).
*   `WithImu/`: Simulations of Extended Kalman Filter (EKF) using synthetic IMU data and handling IMU biases.
*   `ErrorStateKF/`: Implementations of Error-State Extended Kalman Filter (ES-EKF).
*   `TightlyCoupledGPS/`: Advanced simulations demonstrating Tightly Coupled GPS integration, handling satellite dropouts, and comparing Batch vs. Serial updates.
*   `libs/`: Core simulation libraries for GPS (`GPSv1.py`) and IMU (`IMUv1.py`).
*   `DroneSim.py`: A basic interactive drone path simulation.

## Setup Instructions

A bash script is provided to set up a Python virtual environment and install the required dependencies.

1.  Make sure you are in the root directory of the project.
2.  Source the environment initialization script:
    ```bash
    source initialize-environment.sh
    ```
    *Note: You must use `source` (or `.`) so that the virtual environment variables and `PYTHONPATH` are correctly exported to your current shell session.*

## Running the Simulations

After activating the environment, you can run the basic drone simulation:

```bash
python DroneSim.py
```

You can also run any of the other simulation scripts located within the subdirectories. For example:

```bash
python NoImu/EKFvsKFNoImu.py
```
