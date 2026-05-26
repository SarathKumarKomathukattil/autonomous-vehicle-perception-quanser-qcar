'''laptop_server_yolo.py

Server with YOLO detection for traffic lights and stop signs
Run this on Laptop FIRST!
'''

from pal.utilities.stream import BasicStream
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import time
import numpy as np
import cv2
from ultralytics import YOLO

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

# === YOLO MODEL PATH ===
# Update this path to match your system
YOLO_MODEL_PATH = r"C:\Users\kcksa\Documents\Final_Project\runs\detect\unified_traffic_detector\weights\best.pt"

print("=" * 60)
print("=== QCar YOLO Detection Server ===")
print("=" * 60)

# === LOAD YOLO MODEL ===
print(f"\nLoading YOLO model from:\n{YOLO_MODEL_PATH}")
model = YOLO(YOLO_MODEL_PATH)
print("✓ YOLO model loaded!")

# Print class names
print(f"\nDetectable classes: {model.names}")

# === BUFFER ===
imageData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

# === SERVER ===
myServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=imageBufferSize,
    receiveBuffer=imageData,
    nonBlocking=False
)

timeout = Timeout(seconds=2, nanoseconds=0)
prev_con = False

# Control values
throttle = 0.0
steering = 0.0

# Detection state
stop_detected = False
red_light_detected = False

frameCount = 0

print("\n" + "=" * 60)
print("CONTROLS:")
print("  W = Forward    S = Reverse")
print("  A = Left       D = Right")
print("  SPACE = Stop   Q = Quit")
print("")
print("YOLO will automatically stop for:")
print("  - Red traffic lights")
print("  - Stop signs")
print("=" * 60)
print("\nWaiting for QCar...")

try:
    while True:
        
        if not myServer.connected:
            myServer.checkConnection(timeout=timeout)
        
        if myServer.connected and not prev_con:
            print('\n✓ QCar Connected!')
            print('Click on video window to control\n')
        prev_con = myServer.connected
        
        if myServer.connected:
            
            # 1. Receive frame
            recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)
            
            if recvFlag and bytesReceived > 0:
                frameCount += 1
                
                frame = myServer.receiveBuffer.copy()
                
                # 2. Run YOLO detection
                results = model(frame, verbose=False)
                
                # 3. Process detections
                stop_detected = False
                red_light_detected = False
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_id]
                        
                        # Get bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Check for stop conditions
                        if conf > 0.5:  # Confidence threshold
                            # Check class name (adjust based on your training)
                            if 'stop' in cls_name.lower():
                                stop_detected = True
                                color = (0, 0, 255)  # Red
                            elif 'red' in cls_name.lower():
                                red_light_detected = True
                                color = (0, 0, 255)  # Red
                            elif 'green' in cls_name.lower():
                                color = (0, 255, 0)  # Green
                            else:
                                color = (255, 255, 0)  # Cyan
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f'{cls_name} {conf:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 4. Auto-stop if red light or stop sign
                if stop_detected or red_light_detected:
                    throttle = 0.0
                    steering = 0.0
                    status = "STOP SIGN!" if stop_detected else "RED LIGHT!"
                    cv2.putText(frame, f'*** {status} - STOPPING ***', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 5. Display
                cv2.putText(frame, f'Throttle: {throttle:+.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Steering: {steering:+.2f}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Frame: {frameCount}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('QCar YOLO Detection', frame)
                
                # 6. Keyboard input (only if no stop condition)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print('Quitting...')
                    break
                elif not (stop_detected or red_light_detected):
                    # Only allow movement if no stop condition
                    if key == ord('w'):
                        throttle = 0.1
                        steering = 0.0
                        print('>> Forward')
                    elif key == ord('s'):
                        throttle = -0.1
                        steering = 0.0
                        print('>> Reverse')
                    elif key == ord('a'):
                        throttle = 0.05
                        steering = 0.3
                        print('>> Left')
                    elif key == ord('d'):
                        throttle = 0.05
                        steering = -0.3
                        print('>> Right')
                    elif key == ord(' '):
                        throttle = 0.0
                        steering = 0.0
                        print('>> Stop')
                
                # 7. Send control to QCar
                controlData[0] = throttle
                controlData[1] = steering
                myServer.send(controlData.data)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Send stop command
    controlData[0] = 0.0
    controlData[1] = 0.0
    if myServer.connected:
        myServer.send(controlData.data)
        time.sleep(0.1)
    
    myServer.terminate()
    cv2.destroyAllWindows()
    print(f'\nServer closed. Total frames: {frameCount}')