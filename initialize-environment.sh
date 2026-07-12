#/bin/bash

if [ -f KalmanFilter/bin/activate ]; then
	echo "Python environment already setup. Sourcing..."
	if command -v deactivate &> /dev/null; then deactivate; fi
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

export PYTHONPATH=$(pwd)/libs:$PYTHONPATH



