'''laptop_server_white_tuning.py

Tune HSV values for white border detection
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

# === YELLOW HSV (Already tuned - locked) ===
YELLOW_LOWER = np.array([3, 50, 102])
YELLOW_UPPER = np.array([40, 255, 255])

# === WHITE HSV (Initial values - to be tuned) ===
w_h_min, w_s_min, w_v_min = 0, 0, 150
w_h_max, w_s_max, w_v_max = 180, 30, 255

ROI_START_HEIGHT = 0.4

print("=" * 60)
print("=== White Border HSV Tuning ===")
print("=" * 60)
print("\nYellow line detection: LOCKED (already tuned)")
print("Now tuning: White border detection")
print("\nGoal: Detect white/gray border on LEFT side")
print("Avoid: Yellow line, dark road")
print("\nPress 'p' to print values")
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
    """Trackbar callback"""
    pass


def detect_boundaries(frame, white_hsv_lower, white_hsv_upper):
    """Detect yellow (locked) and white (tuning)"""
    h, w = frame.shape[:2]
    roi_start = int(h * ROI_START_HEIGHT)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[roi_start:, :]
    
    # Yellow detection (locked)
    yellow_mask = cv2.inRange(hsv_roi, YELLOW_LOWER, YELLOW_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    yellow_pixels = np.where(yellow_mask > 0)
    yellow_pos = None
    yellow_count = 0
    if len(yellow_pixels[1]) > 0:
        yellow_pos = int(np.mean(yellow_pixels[1]))
        yellow_count = len(yellow_pixels[1])
    
    # White detection (tuning)
    white_mask = cv2.inRange(hsv_roi, white_hsv_lower, white_hsv_upper)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    white_pixels = np.where(white_mask > 0)
    white_pos = None
    white_count = 0
    if len(white_pixels[1]) > 0:
        # Get leftmost white edge
        white_pos = int(np.min(white_pixels[1]))
        white_count = len(white_pixels[1])
    
    # Full masks
    yellow_full = np.zeros((h, w), dtype=np.uint8)
    yellow_full[roi_start:, :] = yellow_mask
    
    white_full = np.zeros((h, w), dtype=np.uint8)
    white_full[roi_start:, :] = white_mask
    
    return yellow_pos, white_pos, yellow_full, white_full, yellow_count, white_count


def draw_visualization(frame, yellow_pos, white_pos, yellow_count, white_count):
    """Draw both boundaries"""
    display = frame.copy()
    h, w = frame.shape[:2]
    roi_y = int(h * ROI_START_HEIGHT)
    
    # ROI box
    cv2.rectangle(display, (0, roi_y), (w, h), (100, 100, 100), 2)
    
    # Frame center (pink)
    center_x = w // 2
    cv2.line(display, (center_x, roi_y), (center_x, h), (255, 0, 255), 2)
    
    # Yellow line (cyan) - LOCKED
    if yellow_pos is not None:
        cv2.line(display, (yellow_pos, roi_y), (yellow_pos, h), (0, 255, 255), 3)
        cv2.putText(display, f'Yellow: {yellow_pos}px (LOCKED)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # White border (white/green) - TUNING
    if white_pos is not None:
        cv2.line(display, (white_pos, roi_y), (white_pos, h), (255, 255, 255), 4)
        cv2.putText(display, f'White Border Detected!', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f'Position: {white_pos}px', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f'Pixels: {white_count}', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Calculate lane center
        if yellow_pos is not None:
            lane_center = int((yellow_pos + white_pos) / 2)
            cv2.line(display, (lane_center, roi_y), (lane_center, h), (0, 255, 0), 3)
            cv2.putText(display, f'Lane Center: {lane_center}px', (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, 'WHITE BORDER NOT DETECTED', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, 'Adjust HSV sliders', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    return display


# Create windows
cv2.namedWindow('QCar - White Border Tuning')
cv2.namedWindow('White HSV Tuning')
cv2.namedWindow('Yellow Mask (Locked)')
cv2.namedWindow('White Mask (Tuning)')

# Create trackbars for WHITE only
cv2.createTrackbar('H Min', 'White HSV Tuning', w_h_min, 179, nothing)
cv2.createTrackbar('S Min', 'White HSV Tuning', w_s_min, 255, nothing)
cv2.createTrackbar('V Min', 'White HSV Tuning', w_v_min, 255, nothing)
cv2.createTrackbar('H Max', 'White HSV Tuning', w_h_max, 179, nothing)
cv2.createTrackbar('S Max', 'White HSV Tuning', w_s_max, 255, nothing)
cv2.createTrackbar('V Max', 'White HSV Tuning', w_v_max, 255, nothing)

try:
    while True:
        
        if not myServer.connected:
            myServer.checkConnection(timeout=timeout)
        
        if myServer.connected and not prev_con:
            print('\n✓ QCar Connected!')
            print('Position QCar where white border is visible')
            print('Adjust WHITE HSV sliders (yellow is locked)\n')
        prev_con = myServer.connected
        
        if myServer.connected:
            
            # Receive frame
            recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)
            
            if recvFlag and bytesReceived > 0:
                frameCount += 1
                
                frame = myServer.receiveBuffer.copy()
                
                # Get WHITE HSV values from trackbars
                w_h_min = cv2.getTrackbarPos('H Min', 'White HSV Tuning')
                w_s_min = cv2.getTrackbarPos('S Min', 'White HSV Tuning')
                w_v_min = cv2.getTrackbarPos('V Min', 'White HSV Tuning')
                w_h_max = cv2.getTrackbarPos('H Max', 'White HSV Tuning')
                w_s_max = cv2.getTrackbarPos('S Max', 'White HSV Tuning')
                w_v_max = cv2.getTrackbarPos('V Max', 'White HSV Tuning')
                
                white_lower = np.array([w_h_min, w_s_min, w_v_min])
                white_upper = np.array([w_h_max, w_s_max, w_v_max])
                
                # Detect both boundaries
                yellow_pos, white_pos, yellow_mask, white_mask, yellow_count, white_count = detect_boundaries(
                    frame, white_lower, white_upper
                )
                
                # Visualize
                display_frame = draw_visualization(frame, yellow_pos, white_pos, yellow_count, white_count)
                
                # Add frame counter
                cv2.putText(display_frame, f'Frame: {frameCount}', (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Show main window
                cv2.imshow('QCar - White Border Tuning', display_frame)
                
                # Show yellow mask (locked - for reference)
                yellow_vis = np.zeros_like(frame)
                yellow_vis[yellow_mask > 0] = [0, 255, 255]  # Cyan
                cv2.imshow('Yellow Mask (Locked)', yellow_vis)
                
                # Show white mask (tuning)
                white_vis = np.zeros_like(frame)
                white_vis[white_mask > 0] = [255, 255, 255]  # White
                cv2.imshow('White Mask (Tuning)', white_vis)
                
                # Show HSV info
                info_img = np.zeros((250, 450, 3), dtype=np.uint8)
                cv2.putText(info_img, 'YELLOW (LOCKED):', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(info_img, f'  [{YELLOW_LOWER[0]}, {YELLOW_LOWER[1]}, {YELLOW_LOWER[2]}] - [{YELLOW_UPPER[0]}, {YELLOW_UPPER[1]}, {YELLOW_UPPER[2]}]', 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.putText(info_img, 'WHITE (TUNING):', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(info_img, f'Lower: [{w_h_min}, {w_s_min}, {w_v_min}]', (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_img, f'Upper: [{w_h_max}, {w_s_max}, {w_v_max}]', (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(info_img, 'Press P to print both values', (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow('White HSV Tuning', info_img)
                
                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print('\nQuitting...')
                    break
                elif key == ord('p'):
                    # Print BOTH values
                    print('\n' + '=' * 60)
                    print('FINAL HSV VALUES FOR BOTH BOUNDARIES:')
                    print('=' * 60)
                    print('\n# Yellow line (right boundary)')
                    print(f'YELLOW_LOWER = np.array([{YELLOW_LOWER[0]}, {YELLOW_LOWER[1]}, {YELLOW_LOWER[2]}])')
                    print(f'YELLOW_UPPER = np.array([{YELLOW_UPPER[0]}, {YELLOW_UPPER[1]}, {YELLOW_UPPER[2]}])')
                    print(f'\n# Yellow detection: {"SUCCESS" if yellow_pos else "FAILED"} ({yellow_count} pixels)')
                    
                    print(f'\n# White border (left boundary)')
                    print(f'WHITE_LOWER = np.array([{w_h_min}, {w_s_min}, {w_v_min}])')
                    print(f'WHITE_UPPER = np.array([{w_h_max}, {w_s_max}, {w_v_max}])')
                    print(f'\n# White detection: {"SUCCESS" if white_pos else "FAILED"} ({white_count} pixels)')
                    print('=' * 60 + '\n')
                
                # Send STOP (no movement)
                controlData[0] = 0.0
                controlData[1] = 0.0
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
    print(f'\nTotal frames: {frameCount}')
    print('White border HSV tuning closed.')