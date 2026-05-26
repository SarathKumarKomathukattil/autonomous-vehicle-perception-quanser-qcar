'''qcar_client_steering_with_lidar.py

QCar Client - Receives control, drives motors, and sends camera + LIDAR data
Run this on QCar!
'''

from pal.utilities.stream import BasicStream
from pal.products.qcar import QCarCameras, QCar
from pal.utilities.lidar import Lidar
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import time
import numpy as np
import cv2

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

# LIDAR parameters
numMeasurements = 360
lidarBufferSize = numMeasurements * 4 * 2  # 360 points * 4 bytes/float * 2 (angles + distances)

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_IP = '192.168.2.20'
SERVER_PORT = 18001
LIDAR_PORT = 18002  # Separate port for LIDAR data

print("=" * 60)
print("=== QCar Steering + LIDAR Test - Client ===")
print("=" * 60)

# === INITIALIZE CAMERA ===
print("\nInitializing camera...")
cameras = QCarCameras(
    enableFront=True,
    enableBack=False,
    enableLeft=False,
    enableRight=False
)
print("✓ Camera ready")

# === INITIALIZE LIDAR ===
print("Initializing LIDAR...")
try:
    myLidar = Lidar(
        type='RPLidar',
        numMeasurements=numMeasurements,
        rangingDistanceMode=2,  # Standard mode
        interpolationMode=0
    )
    print("✓ LIDAR ready")
    print(f"  LIDAR will send {numMeasurements} measurements")
    lidar_enabled = True
except Exception as e:
    print(f"⚠ LIDAR initialization failed: {e}")
    print("  Continuing without LIDAR...")
    lidar_enabled = False

# === INITIALIZE MOTORS ===
print("Initializing motors...")
myCar = QCar(readMode=1, frequency=100)
print("✓ Motors ready")

# === BUFFERS ===
imageData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)
lidarData = np.zeros(numMeasurements * 2, dtype=np.float32)  # angles + distances interleaved

# === CLIENT STREAMS ===
print(f"\nConnecting to {SERVER_IP}:{SERVER_PORT}...")
myClient = BasicStream(
    f'tcpip://{SERVER_IP}:{SERVER_PORT}',
    agent='C',
    sendBufferSize=imageBufferSize,
    recvBufferSize=controlBufferSize,
    receiveBuffer=controlData,
    nonBlocking=False
)
print("✓ Camera stream ready")

# LIDAR stream (separate connection)
if lidar_enabled:
    print(f"Connecting LIDAR stream to {SERVER_IP}:{LIDAR_PORT}...")
    myLidarClient = BasicStream(
        f'tcpip://{SERVER_IP}:{LIDAR_PORT}',
        agent='C',
        sendBufferSize=lidarBufferSize,
        recvBufferSize=0,
        nonBlocking=False
    )
    print("✓ LIDAR stream ready")

timeout = Timeout(seconds=2, nanoseconds=0)
recvTimeout = Timeout(seconds=0, nanoseconds=100000000)  # 100ms
prev_con = False
prev_lidar_con = False

# Control values
throttle = 0.0
steering = 0.0

# LEDs array
LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

frameCount = 0
lidarCount = 0
lidar_error_count = 0

print("\nWaiting for server...")
print("Press 'q' on QCar to quit\n")

try:
    while True:
        
        # Check camera stream connection
        if not myClient.connected:
            myClient.checkConnection(timeout=timeout)
            # Safety: stop motors when not connected
            myCar.write(0, 0, LEDs)
        
        if myClient.connected and not prev_con:
            print('✓ Connected to Server (Camera)!')
        prev_con = myClient.connected
        
        # Check LIDAR stream connection
        if lidar_enabled:
            if not myLidarClient.connected:
                myLidarClient.checkConnection(timeout=timeout)
            
            if myLidarClient.connected and not prev_lidar_con:
                print('✓ Connected to Server (LIDAR)!')
                print('  Starting LIDAR data transmission...\n')
            prev_lidar_con = myLidarClient.connected
        
        if myClient.connected:
            
            # 1. Read sensors
            myCar.read()
            
            # 2. Send camera frame
            cameras.readAll()
            frame = cameras.csiFront.imageData
            
            if frame.shape[0] != imageHeight or frame.shape[1] != imageWidth:
                frame = cv2.resize(frame, (imageWidth, imageHeight))
            
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            np.copyto(imageData, frame)
            myClient.send(imageData.data)
            frameCount += 1
            
            # 3. Read and send LIDAR data
            if lidar_enabled and myLidarClient.connected:
                try:
                    myLidar.read()
                    
                    # Debug: Print first read to verify data
                    if lidarCount == 0:
                        print(f"\n📡 First LIDAR read:")
                        print(f"  Angles: min={np.min(myLidar.angles):.3f}, max={np.max(myLidar.angles):.3f}")
                        print(f"  Distances: min={np.min(myLidar.distances):.3f}, max={np.max(myLidar.distances):.3f}")
                        print(f"  Non-zero distances: {np.count_nonzero(myLidar.distances)}/{numMeasurements}")
                    
                    # Verify we have valid data
                    if np.count_nonzero(myLidar.distances) > 0:
                        # Interleave angles and distances for transmission
                        # Format: [angle1, dist1, angle2, dist2, ...]
                        for i in range(numMeasurements):
                            lidarData[i * 2] = myLidar.angles[i]
                            lidarData[i * 2 + 1] = myLidar.distances[i]
                        
                        myLidarClient.send(lidarData.data)
                        lidarCount += 1
                        
                        # Periodic debug info
                        if lidarCount % 100 == 0:
                            valid_pts = np.sum((myLidar.distances > 0.15) & (myLidar.distances < 8.0))
                            print(f"📡 LIDAR: {lidarCount} scans sent | Valid pts: {valid_pts}/360")
                    else:
                        lidar_error_count += 1
                        if lidar_error_count % 50 == 0:
                            print(f"⚠ LIDAR returning all zeros ({lidar_error_count} times)")
                    
                except Exception as e:
                    if frameCount % 100 == 0:
                        print(f"⚠ LIDAR read error: {e}")
            
            # 4. Receive control from server
            recvFlag, bytesReceived = myClient.receive(timeout=recvTimeout)
            
            if recvFlag and bytesReceived == controlBufferSize:
                throttle = float(myClient.receiveBuffer[0])
                steering = float(myClient.receiveBuffer[1])
            
            # 5. Update LEDs based on steering/throttle
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
            if steering > 0.15:
                LEDs[0] = 1
                LEDs[2] = 1
            elif steering < -0.15:
                LEDs[1] = 1
                LEDs[3] = 1
            if throttle < 0:
                LEDs[5] = 1
            
            # 6. Write to motors
            myCar.write(throttle, steering, LEDs)
            
            # 7. Print status
            if frameCount % 30 == 0:
                lidar_status = f'LIDAR: {lidarCount}' if lidar_enabled else 'LIDAR: OFF'
                print(f'Frame {frameCount} | {lidar_status} | T: {throttle:+.2f} | S: {steering:+.2f} | Bat: {myCar.batteryVoltage:.1f}V')
            
            # 8. Display locally
            cv2.putText(frame, f'T: {throttle:+.2f} S: {steering:+.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if lidar_enabled:
                cv2.putText(frame, f'LIDAR: {lidarCount}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('QCar Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('\nUser pressed q to quit')
                break

except KeyboardInterrupt:
    print("\n\nUser interrupted with Ctrl+C")

finally:
    # SAFETY: Stop motors!
    print("\n" + "=" * 40)
    print("Stopping motors...")
    LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    myCar.write(0, 0, LEDs)
    time.sleep(0.2)
    
    print("Closing connections...")
    cameras.terminate()
    myCar.terminate()
    myClient.terminate()
    
    if lidar_enabled:
        myLidar.terminate()
        myLidarClient.terminate()
    
    cv2.destroyAllWindows()
    
    print("=" * 40)
    print(f'Total frames: {frameCount}')
    if lidar_enabled:
        print(f'Total LIDAR scans: {lidarCount}')
        if lidar_error_count > 0:
            print(f'LIDAR errors: {lidar_error_count}')
    print('QCar client closed safely.')
    print("=" * 40)