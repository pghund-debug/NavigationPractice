#/bin/bash

if [ -f KalmanFilter/bin/activate ]; then
	echo "Python environment already setup. Sourcing..."
	source KalmanFilter/bin/activate
else
	echo "No python environment found. Creating..."
	rm -rf KalmanFilter/
	python3 -m venv KalmanFilter
	source KalmanFilter/bin/activate
	pip install --upgrade pip
	pip install contourpy
	pip install numpy
	pip install scipy
	pip install pillow
	pip install matplotlib
	pip install pyserial
fi



