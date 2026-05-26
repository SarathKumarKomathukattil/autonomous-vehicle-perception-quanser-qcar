'''laptop_server_data_collector_v2.py

Collect images from QCar for segmentation training
- Supports W+A, W+D combinations!
- A = LEFT, D = RIGHT
- Very slow speed
- Release = Stop
'''

from pal.utilities.stream import BasicStream
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import time
import numpy as np
import cv2
import os

# Try to import keyboard library for multi-key support
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
    print("✓ Keyboard library available - Multi-key support enabled!")
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠ Install 'keyboard' library for W+A/W+D support: pip install keyboard")
    print("  Using single-key mode for now...")

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

# === SPEED SETTINGS (VERY SLOW) ===
FORWARD_SPEED = 0.05   # Very slow forward
REVERSE_SPEED = -0.035   # Very slow reverse
STEERING_AMOUNT = 0.30   # Steering angle

# === DATA COLLECTION SETTINGS ===
SAVE_DIR = 'training_data/images'
AUTO_CAPTURE_INTERVAL = 0.5

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("=== QCar Data Collection (Multi-Key Support) ===")
print("=" * 60)

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

# Data collection state
image_count = 0
auto_capture = False
last_capture_time = 0

frameCount = 0

# Count existing images
existing_images = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
image_count = existing_images
print(f"\nExisting images in folder: {existing_images}")

print("\n" + "=" * 60)
print("DRIVING CONTROLS:")
print("  W     = Forward")
print("  S     = Reverse")
print("  A     = Steer LEFT")
print("  D     = Steer RIGHT")
print("  W + A = Forward + Left")
print("  W + D = Forward + Right")
print("  S + A = Reverse + Left")
print("  S + D = Reverse + Right")
print("  Release all = STOP")
print("")
print("OTHER CONTROLS:")
print("  + / - = Adjust speed")
print("  C     = Capture image")
print("  R     = Toggle auto-capture")
print("  Q     = Quit")
print("=" * 60)
print(f"\nSpeed: {FORWARD_SPEED} | Steering: {STEERING_AMOUNT}")
print(f"Save folder: {SAVE_DIR}")
print("\nWaiting for QCar...")


def get_keyboard_state():
    """Get current state of WASD keys"""
    if KEYBOARD_AVAILABLE:
        return {
            'w': keyboard.is_pressed('w'),
            'a': keyboard.is_pressed('a'),
            's': keyboard.is_pressed('s'),
            'd': keyboard.is_pressed('d')
        }
    else:
        return {'w': False, 'a': False, 's': False, 'd': False}


def calculate_control(keys):
    """
    Calculate throttle and steering from key states
    A = LEFT (positive steering)
    D = RIGHT (negative steering)
    """
    throttle = 0.0
    steering = 0.0
    
    # Throttle
    if keys['w']:
        throttle = FORWARD_SPEED
    elif keys['s']:
        throttle = REVERSE_SPEED
    
    # Steering - A=LEFT, D=RIGHT
    if keys['a']:
        steering = STEERING_AMOUNT   # Positive = LEFT
    elif keys['d']:
        steering = -STEERING_AMOUNT  # Negative = RIGHT
    
    return throttle, steering


try:
    while True:
        
        if not myServer.connected:
            myServer.checkConnection(timeout=timeout)
        
        if myServer.connected and not prev_con:
            print('\n✓ QCar Connected!')
            print('Use WASD to drive (W+A for forward-left, etc.)')
            print('Press C to capture, R for auto-capture\n')
        prev_con = myServer.connected
        
        if myServer.connected:
            
            # 1. Receive frame
            recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)
            
            if recvFlag and bytesReceived > 0:
                frameCount += 1
                
                frame = myServer.receiveBuffer.copy()
                
                # 2. Get keyboard state (supports multiple keys!)
                if KEYBOARD_AVAILABLE:
                    keys = get_keyboard_state()
                    throttle, steering = calculate_control(keys)
                
                # 3. Auto-capture logic
                current_time = time.time()
                if auto_capture and (current_time - last_capture_time) >= AUTO_CAPTURE_INTERVAL:
                    filename = f'{SAVE_DIR}/frame_{image_count:04d}.jpg'
                    cv2.imwrite(filename, frame)
                    image_count += 1
                    last_capture_time = current_time
                    print(f'📸 Auto: frame_{image_count-1:04d}.jpg ({image_count} total)')
                
                # 4. Display
                display = frame.copy()
                
                # Top status bar
                cv2.rectangle(display, (0, 0), (imageWidth, 100), (0, 0, 0), -1)
                
                # Control values
                cv2.putText(display, f'Throttle: {throttle:+.3f}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f'Steering: {steering:+.3f}', (220, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f'Speed: {FORWARD_SPEED:.3f}', (430, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Image counter
                cv2.putText(display, f'Images: {image_count}', (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Movement status
                if KEYBOARD_AVAILABLE:
                    keys = get_keyboard_state()
                    if keys['w'] or keys['s']:
                        if keys['a']:
                            direction = 'FWD+LEFT' if keys['w'] else 'REV+LEFT'
                        elif keys['d']:
                            direction = 'FWD+RIGHT' if keys['w'] else 'REV+RIGHT'
                        else:
                            direction = 'FORWARD' if keys['w'] else 'REVERSE'
                        cv2.putText(display, direction, (220, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, 'STOPPED', (220, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show pressed keys
                    keys_str = ''
                    if keys['w']: keys_str += 'W '
                    if keys['a']: keys_str += 'A '
                    if keys['s']: keys_str += 'S '
                    if keys['d']: keys_str += 'D '
                    if keys_str:
                        cv2.putText(display, f'Keys: [{keys_str.strip()}]', (10, 82),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    else:
                        cv2.putText(display, f'Frame: {frameCount}', (10, 82),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                # Recording indicator
                if auto_capture:
                    cv2.circle(display, (imageWidth - 30, 50), 15, (0, 0, 255), -1)
                    cv2.putText(display, 'REC', (imageWidth - 80, 57),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Bottom instructions
                cv2.rectangle(display, (0, imageHeight - 30), (imageWidth, imageHeight), (0, 0, 0), -1)
                cv2.putText(display, 'WASD=Drive | C=Capture | R=Record | +/-=Speed | Q=Quit', 
                            (10, imageHeight - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                cv2.imshow('QCar Data Collection', display)
                
                # 5. Handle special keys via OpenCV
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print('\nQuitting...')
                    break
                elif key == ord('c'):
                    filename = f'{SAVE_DIR}/frame_{image_count:04d}.jpg'
                    cv2.imwrite(filename, frame)
                    image_count += 1
                    print(f'📸 Captured: frame_{image_count-1:04d}.jpg ({image_count} total)')
                elif key == ord('r'):
                    auto_capture = not auto_capture
                    if auto_capture:
                        print('\n🔴 AUTO-CAPTURE ON')
                        last_capture_time = time.time()
                    else:
                        print('\n⬜ AUTO-CAPTURE OFF')
                elif key == ord('+') or key == ord('='):
                    FORWARD_SPEED = min(FORWARD_SPEED + 0.005, 0.08)
                    REVERSE_SPEED = -FORWARD_SPEED
                    print(f'Speed: {FORWARD_SPEED:.3f}')
                elif key == ord('-') or key == ord('_'):
                    FORWARD_SPEED = max(FORWARD_SPEED - 0.005, 0.025)
                    REVERSE_SPEED = -FORWARD_SPEED
                    print(f'Speed: {FORWARD_SPEED:.3f}')
                
                # Fallback for single-key mode (if keyboard library not available)
                if not KEYBOARD_AVAILABLE:
                    throttle = 0.0
                    steering = 0.0
                    if key == ord('w'):
                        throttle = FORWARD_SPEED
                    elif key == ord('s'):
                        throttle = REVERSE_SPEED
                    elif key == ord('a'):
                        throttle = FORWARD_SPEED
                        steering = STEERING_AMOUNT
                    elif key == ord('d'):
                        throttle = FORWARD_SPEED
                        steering = -STEERING_AMOUNT
                
                # 6. Send control to QCar
                controlData[0] = throttle
                controlData[1] = steering
                myServer.send(controlData.data)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    controlData[0] = 0.0
    controlData[1] = 0.0
    if myServer.connected:
        myServer.send(controlData.data)
        time.sleep(0.1)
    
    myServer.terminate()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total images saved: {image_count}")
    print(f"Save location: {os.path.abspath(SAVE_DIR)}")
    print("=" * 60)
    
    if image_count > 0:
        print("\nNEXT STEPS:")
        print("1. pip install labelme")
        print(f"2. labelme {SAVE_DIR}")
        print("3. Label: road, yellow_line, white_line, roundabout")