import serial
import time

# --- Configuration ---
PORT = '/dev/ttyACM0'         # Change to your Arduino's port (e.g., '/dev/ttyACM0' on Linux)
BAUD_RATE = 9600    # Match the Serial.begin(XXXX) speed in your Arduino sketch
OUTPUT_FILE = "imu_raw_data.txt"
COLLECTION_TIME_SECS = 3600 * 5  # Duration for testing (Set to 7200 for a 2-hour run!)

print(f"Opening port {PORT}...")
try:
    # Open the serial port
    # timeout=1 ensures the script doesn't freeze forever if the Arduino stops sending data
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Crucial: Arduino resets when serial opens. Wait 2 seconds for it to boot.
    ser.reset_input_buffer() # Flush any old, corrupted partial data strings
    print("Connected successfully!")
except Exception as e:
    print(f"Error opening port: {e}")
    exit()

print(f"Recording data to '{OUTPUT_FILE}'... Press Ctrl+C to stop early.")

start_time = time.time()
lines_recorded = 0

with open(OUTPUT_FILE, "w") as f:
    try:
        while time.time() - start_time < COLLECTION_TIME_SECS:
            if ser.in_waiting > 0:
                # Read a single line of text ending in '\n'
                # .decode('utf-8') turns bytes into a standard python string
                # .strip() removes trailing whitespaces/newlines
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line:  # Check if we got valid text
                    # Print to console so you can see it working in real-time
                   # print(line) 
                    
                    # Write to your text file
                    f.write(line + "\n")
                    lines_recorded += 1
                    
    except KeyboardInterrupt:
        print("\nRecording stopped early by user.")

# Clean up
ser.close()
print(f"\nFinished! Recorded {lines_recorded} lines to '{OUTPUT_FILE}'. Port closed safely.")
