'''laptop_server_hsv_tuning.py

HSV tuning for yellow line detection
No motor control - QCar stays stationary
Based on your working server code
'''

from pal.utilities.stream import BasicStream
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

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

# === INITIAL HSV VALUES ===
h_min, s_min, v_min = 20, 100, 100
h_max, s_max, v_max = 35, 255, 255

ROI_START_HEIGHT = 0.4  # Focus on bottom 60%

print("=" * 60)
print("=== HSV Yellow Line Tuning Mode ===")
print("=" * 60)
print("\nQCar will NOT move - stationary testing only")
print("Use sliders in 'HSV Tuning' window to adjust detection")
print("\nPress 'p' to print current HSV values")
print("Press 'q' to quit")
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

frameCount = 0

print("\nWaiting for QCar...")


def nothing(x):
    """Trackbar callback (required but does nothing)"""
    pass


def detect_yellow_line(frame, hsv_lower, hsv_upper):
    """
    Detect yellow line using HSV filtering
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply ROI (bottom part of image)
    h, w = frame.shape[:2]
    roi_start = int(h * ROI_START_HEIGHT)
    hsv_roi = hsv[roi_start:, :]
    
    # Create mask for yellow
    yellow_mask = cv2.inRange(hsv_roi, hsv_lower, hsv_upper)
    
    # Clean up mask (remove noise)
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    # Find yellow pixels
    yellow_pixels = np.where(yellow_mask > 0)
    
    yellow_position = None
    pixel_count = 0
    
    if len(yellow_pixels[1]) > 0:
        # Calculate average X position
        yellow_position = int(np.mean(yellow_pixels[1]))
        pixel_count = len(yellow_pixels[1])
    
    # Create full-size mask for display
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_start:, :] = yellow_mask
    
    return full_mask, yellow_position, pixel_count


def draw_visualization(frame, yellow_mask, yellow_position, pixel_count):
    """
    Draw visualization overlay
    """
    display = frame.copy()
    h, w = frame.shape[:2]
    roi_y = int(h * ROI_START_HEIGHT)
    
    # Draw ROI box
    cv2.rectangle(display, (0, roi_y), (w, h), (100, 100, 100), 2)
    cv2.putText(display, 'Detection ROI', (10, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
    
    # Draw frame center reference
    center_x = w // 2
    cv2.line(display, (center_x, roi_y), (center_x, h), (255, 0, 255), 2)
    cv2.putText(display, 'Center', (center_x + 10, roi_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Draw yellow line if detected
    if yellow_position is not None:
        cv2.line(display, (yellow_position, roi_y), 
                (yellow_position, h), (0, 255, 255), 4)
        
        # Show position and pixel count
        cv2.putText(display, f'Yellow Line Detected!', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f'Position: {yellow_position}px', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f'Pixels: {pixel_count}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show offset from center
        offset = yellow_position - center_x
        cv2.putText(display, f'Offset: {offset:+d}px', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display, 'NO YELLOW LINE DETECTED', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display, 'Adjust HSV sliders', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return display


# Create windows
cv2.namedWindow('QCar Camera - Yellow Detection')
cv2.namedWindow('HSV Tuning')
cv2.namedWindow('Yellow Mask')

# Create trackbars for HSV tuning
cv2.createTrackbar('H Min', 'HSV Tuning', h_min, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Tuning', s_min, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Tuning', v_min, 255, nothing)
cv2.createTrackbar('H Max', 'HSV Tuning', h_max, 179, nothing)
cv2.createTrackbar('S Max', 'HSV Tuning', s_max, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Tuning', v_max, 255, nothing)

try:
    while True:
        
        # Check connection
        if not myServer.connected:
            myServer.checkConnection(timeout=timeout)
        
        if myServer.connected and not prev_con:
            print('\n✓ QCar Connected!')
            print('Position QCar on track with yellow line visible')
            print('Adjust HSV sliders until yellow line is detected\n')
        prev_con = myServer.connected
        
        if myServer.connected:
            
            # 1. Receive frame from QCar
            recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)
            
            if recvFlag and bytesReceived > 0:
                frameCount += 1
                
                frame = myServer.receiveBuffer.copy()
                
                # 2. Get current HSV values from trackbars
                h_min = cv2.getTrackbarPos('H Min', 'HSV Tuning')
                s_min = cv2.getTrackbarPos('S Min', 'HSV Tuning')
                v_min = cv2.getTrackbarPos('V Min', 'HSV Tuning')
                h_max = cv2.getTrackbarPos('H Max', 'HSV Tuning')
                s_max = cv2.getTrackbarPos('S Max', 'HSV Tuning')
                v_max = cv2.getTrackbarPos('V Max', 'HSV Tuning')
                
                hsv_lower = np.array([h_min, s_min, v_min])
                hsv_upper = np.array([h_max, s_max, v_max])
                
                # 3. Detect yellow line
                yellow_mask, yellow_position, pixel_count = detect_yellow_line(
                    frame, hsv_lower, hsv_upper
                )
                
                # 4. Create visualizations
                display_frame = draw_visualization(
                    frame, yellow_mask, yellow_position, pixel_count
                )
                
                # Add frame counter
                cv2.putText(display_frame, f'Frame: {frameCount}', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 5. Show main camera view
                cv2.imshow('QCar Camera - Yellow Detection', display_frame)
                
                # 6. Show yellow mask (colorized for visibility)
                mask_color = np.zeros_like(frame)
                mask_color[yellow_mask > 0] = [0, 255, 255]  # Yellow
                cv2.imshow('Yellow Mask', mask_color)
                
                # 7. Show HSV values on tuning window
                info_img = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(info_img, 'Current HSV Range:', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_img, f'Lower: [{h_min}, {s_min}, {v_min}]', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(info_img, f'Upper: [{h_max}, {s_max}, {v_max}]', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(info_img, 'Press P to print values', (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow('HSV Tuning', info_img)
                
                # 8. Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print('\nQuitting...')
                    break
                elif key == ord('p'):
                    # Print current HSV values
                    print('\n' + '=' * 50)
                    print('CURRENT HSV VALUES:')
                    print('=' * 50)
                    print(f'YELLOW_LOWER = np.array([{h_min}, {s_min}, {v_min}])')
                    print(f'YELLOW_UPPER = np.array([{h_max}, {s_max}, {v_max}])')
                    print('=' * 50)
                    if yellow_position is not None:
                        print(f'Detection: SUCCESS ({pixel_count} pixels)')
                    else:
                        print('Detection: FAILED (no yellow pixels)')
                    print('=' * 50 + '\n')
                
                # 9. Send STOP command (no movement)
                controlData[0] = 0.0  # Throttle = 0
                controlData[1] = 0.0  # Steering = 0
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
    print(f'\nTotal frames processed: {frameCount}')
    print('HSV tuning session closed.')