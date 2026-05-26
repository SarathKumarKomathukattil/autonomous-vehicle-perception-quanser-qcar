'''laptop_server_lane_autonomous_with_traffic_light_detection_and_lidar.py

Autonomous Lane Following + UNIFIED Traffic Detection + SIMPLE LIDAR
- LIDAR detects ALL obstacles from 0.1m to 0.6m (no lane filtering)
- VISUAL FEEDBACK ONLY - no stopping yet
'''

from pal.utilities.stream import BasicStream
try:
    from quanser.common import Timeout
except:
    from quanser.communications import Timeout
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

# LIDAR parameters
numMeasurements = 360
lidarBufferSize = numMeasurements * 4 * 2
lidarData = np.zeros(numMeasurements * 2, dtype=np.float32)

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001
LIDAR_PORT = 18002

# === MODEL PATHS ===
LANE_SEG_MODEL_PATH = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data\lane_segmentation_model.pth'
UNIFIED_DETECTION_MODEL_PATH = r'C:\Users\kcksa\Documents\Traffic lights and signs\unified_traffic_detection\exp22\weights\best.pt'

# === SEGMENTATION PARAMETERS ===
NUM_CLASSES = 3
SEG_IMAGE_SIZE = (256, 256)
ROI_START_HEIGHT = 0.2
ROI_END_HEIGHT = 0.9
BOTTOM_PORTION = 0.3

# === CONTROL PARAMETERS ===
BASE_THROTTLE = 0.06
YELLOW_THROTTLE_REDUCTION = 0.10
Kp = -0.003
MAX_STEERING = 0.5
FRAME_CENTER = imageWidth // 2

# === SAFETY OFFSETS ===
ROUNDABOUT_OFFSET = 40
YELLOW_OFFSET = 50
MIN_PIXELS = 2500

# === UNIFIED DETECTION PARAMETERS ===
DETECTION_CONF = 0.3
CHECK_DETECTION_EVERY_N_FRAMES = 1
YOLO_ROI_HEIGHT = 0.5

# === TRAFFIC LIGHT HEIGHT PARAMETERS ===
MIN_HEIGHT_THRESHOLD = 7
STOP_HEIGHT_THRESHOLD = 9
MAX_HEIGHT_THRESHOLD = 11

# === STOP SIGN PARAMETERS ===
STOP_SIGN_MIN_HEIGHT = 40
STOP_SIGN_STOP_HEIGHT = 45
STOP_SIGN_MAX_HEIGHT = 50
STOP_SIGN_DURATION = 3.0

# === LIDAR PARAMETERS ===
LIDAR_MAX_DISTANCE = 0.6
LIDAR_MIN_DISTANCE = 0.1

# === OVERLAY SETTINGS ===
OVERLAY_ALPHA = 0.5
YELLOW_LINE_COLOR = (0, 255, 255)
ROUNDABOUT_COLOR = (255, 0, 255)

DETECTION_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'stop': (0, 0, 255),
    'speed_60': (0, 165, 255),
    'no_right_turn': (255, 0, 255)
}

print("=" * 60)
print("=== AUTONOMOUS + UNIFIED DETECTION + SIMPLE LIDAR ===")
print("=" * 60)

# === CREATE SERVERS ===
imageData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

myServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=imageBufferSize,
    receiveBuffer=imageData,
    nonBlocking=False
)

myLidarServer = BasicStream(
    f'tcpip://0.0.0.0:{LIDAR_PORT}',
    agent='S',
    sendBufferSize=0,
    recvBufferSize=lidarBufferSize,
    receiveBuffer=lidarData,
    nonBlocking=False
)

timeout = Timeout(seconds=2, nanoseconds=0)

print(f"✅ Camera server started on port {SERVER_PORT}")
print(f"✅ LIDAR server started on port {LIDAR_PORT}")
print("⏳ Waiting for QCar connection...")

while not myServer.connected:
    myServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)
print("\n✅ QCar Camera Connected!")

print("⏳ Waiting for LIDAR connection...")
while not myLidarServer.connected:
    myLidarServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)
print("\n✅ QCar LIDAR Connected!")

# === LIDAR DATA STORAGE ===
lidar_angles = np.zeros(numMeasurements, dtype=np.float32)
lidar_distances = np.zeros(numMeasurements, dtype=np.float32)
lidar_frame_count = 0

# === MATPLOTLIB SETUP ===
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_ylim(0, LIDAR_MAX_DISTANCE)
ax.set_title('LIDAR View', pad=20)
canvas = FigureCanvasAgg(fig)

# === LOAD MODELS ===
print("\nLoading lane segmentation model...")

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        self.bottleneck = UNetBlock(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.out = nn.Conv2d(64, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    lane_model = UNet(n_classes=NUM_CLASSES).to(device)
    lane_model.load_state_dict(torch.load(LANE_SEG_MODEL_PATH, map_location=device))
    lane_model.eval()
    print("✅ Lane segmentation model loaded!")
except Exception as e:
    print(f"❌ Could not load lane model: {e}")
    lane_model = None

print("\nLoading UNIFIED detection model...")
try:
    unified_model = YOLO(UNIFIED_DETECTION_MODEL_PATH)
    print("✅ Unified detection model loaded!")
except Exception as e:
    print(f"❌ Could not load unified model: {e}")
    unified_model = None

print("\n" + "=" * 60)
print("✅ READY - AUTONOMOUS + SIMPLE LIDAR!")
print(f"   LIDAR Range: {LIDAR_MIN_DISTANCE}m to {LIDAR_MAX_DISTANCE}m (ALL directions)")
print(f"   🔍 TESTING MODE: Visual feedback only")
print("=" * 60)

# === STATE ===
throttle = 0.0
steering = 0.0
frameCount = 0
calibrated = False
lane_distance = None
emergency_stop = False
yellow_count = 0
round_count = 0
bottom_roi_y = 0
all_detections = []
show_overlay = True
stopped_at_red_light = False
stopped_at_stop_sign = False
stop_sign_start_time = None
show_lidar = True

print("\nCONTROLS: Q=Quit, SPACE=Stop, V=Overlay, L=LIDAR window")
print("=" * 60)


def process_lidar_data():
    """Process LIDAR data"""
    global lidar_angles, lidar_distances, lidar_frame_count
    
    try:
        recv_buffer = myLidarServer.receiveBuffer
        for i in range(numMeasurements):
            lidar_angles[i] = recv_buffer[i * 2]
            lidar_distances[i] = recv_buffer[i * 2 + 1]
        lidar_frame_count += 1
        
        if lidar_frame_count == 1:
            print(f"\n📡 LIDAR: {np.count_nonzero(lidar_distances)}/360 points")
    except Exception as e:
        print(f"⚠ LIDAR error: {e}")


def filter_lidar():
    """Simple filter: all points between 0.1m and 0.6m"""
    return (lidar_distances > LIDAR_MIN_DISTANCE) & (lidar_distances < LIDAR_MAX_DISTANCE)


def draw_lidar_plot():
    """Draw LIDAR visualization"""
    global ax, canvas
    
    ax.clear()
    
    valid_mask = filter_lidar()
    valid_angles = lidar_angles[valid_mask]
    valid_distances = lidar_distances[valid_mask]
    
    if len(valid_distances) == 0:
        ax.text(0, LIDAR_MAX_DISTANCE/2, 'NO OBSTACLES', 
                ha='center', va='center', fontsize=14, color='green', weight='bold')
        ax.set_title(f'LIDAR Frame {lidar_frame_count}: Clear', pad=20, fontsize=10, color='green')
    else:
        # Color by distance
        colors = ['red' if d < 0.2 else 'orange' if d < 0.4 else 'yellow' for d in valid_distances]
        ax.scatter(valid_angles, valid_distances, c=colors, marker='o', s=80, alpha=0.9, edgecolors='black', linewidths=2)
        
        # Closest obstacle
        min_dist = valid_distances.min()
        min_angle = valid_angles[np.argmin(valid_distances)]
        ax.plot([min_angle, min_angle], [0, min_dist], 'r--', linewidth=3, alpha=0.8)
        
        title_color = 'red' if min_dist < 0.2 else 'orange' if min_dist < 0.4 else 'green'
        ax.set_title(f'OBSTACLE: {min_dist:.2f}m | {len(valid_distances)} pts', 
                     pad=20, fontsize=11, fontweight='bold', color=title_color)
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, LIDAR_MAX_DISTANCE)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_rlabel_position(45)
    
    canvas.draw()
    buf = canvas.buffer_rgba()
    lidar_img = np.frombuffer(buf, dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(lidar_img, cv2.COLOR_RGBA2BGR)


def detect_all_objects(frame):
    """Detect traffic objects"""
    if unified_model is None:
        return []
    
    yolo_roi_end = int(imageHeight * YOLO_ROI_HEIGHT)
    results = unified_model(frame[0:yolo_roi_end, :], conf=DETECTION_CONF, verbose=False)
    
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'class': unified_model.names[int(box.cls[0])],
                'conf': float(box.conf[0]),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'height': int(y2 - y1)
            })
    return detections


def check_red_light(detections):
    global stopped_at_red_light
    for d in detections:
        if d['class'] == 'red' and MIN_HEIGHT_THRESHOLD <= d['height'] <= MAX_HEIGHT_THRESHOLD:
            if not stopped_at_red_light:
                return d['height'] >= STOP_HEIGHT_THRESHOLD
            return True
    return False


def check_yellow_light(detections):
    for d in detections:
        if d['class'] == 'yellow' and MIN_HEIGHT_THRESHOLD <= d['height'] <= MAX_HEIGHT_THRESHOLD:
            return True
    return False


def check_green_light(detections):
    for d in detections:
        if d['class'] == 'green' and MIN_HEIGHT_THRESHOLD <= d['height'] <= MAX_HEIGHT_THRESHOLD:
            return True
    return False


def check_stop_sign(detections):
    global stopped_at_stop_sign
    for d in detections:
        if d['class'] == 'stop' and STOP_SIGN_MIN_HEIGHT <= d['height'] <= STOP_SIGN_MAX_HEIGHT:
            if not stopped_at_stop_sign:
                return d['height'] >= STOP_SIGN_STOP_HEIGHT
            return True
    return False


def segment_lane(frame):
    global yellow_count, round_count, bottom_roi_y
    
    if lane_model is None:
        return None, None, None

    img_resized = cv2.resize(frame, SEG_IMAGE_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = lane_model(img_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    seg_mask = cv2.resize(prediction.astype(np.uint8), (imageWidth, imageHeight), interpolation=cv2.INTER_NEAREST)
    roi_start = int(imageHeight * ROI_START_HEIGHT)
    roi_end = int(imageHeight * ROI_END_HEIGHT)
    bottom_start = roi_start + int((roi_end - roi_start) * BOTTOM_PORTION)
    bottom_roi_y = bottom_start
    bottom_roi_mask = seg_mask[bottom_start:roi_end, :]

    yellow_pixels = np.where(bottom_roi_mask == 1)
    yellow_count = len(yellow_pixels[1])
    yellow_pos = int(np.mean(yellow_pixels[1])) if yellow_count > MIN_PIXELS else None

    round_pixels = np.where(bottom_roi_mask == 2)
    round_count = len(round_pixels[1])
    round_pos = int(np.mean(round_pixels[1])) if round_count > MIN_PIXELS else None

    return seg_mask, yellow_pos, round_pos


def calculate_target(yellow_pos, round_pos, lane_distance):
    if yellow_pos is not None and round_pos is not None:
        return (round_pos + ROUNDABOUT_OFFSET + yellow_pos + YELLOW_OFFSET) // 2, "DRIVING"
    elif yellow_pos is not None and lane_distance is not None:
        return int(yellow_pos - lane_distance + YELLOW_OFFSET), "YELLOW ONLY"
    elif round_pos is not None and lane_distance is not None:
        return int(round_pos + lane_distance + ROUNDABOUT_OFFSET), "ROUND ONLY"
    return None, "NO LANE"


def draw_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, detections):
    display = frame.copy()
    h, w = frame.shape[:2]

    if seg_mask is not None and show_overlay:
        overlay = display.copy()
        overlay[seg_mask == 1] = YELLOW_LINE_COLOR
        overlay[seg_mask == 2] = ROUNDABOUT_COLOR
        display = cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1 - OVERLAY_ALPHA, 0)

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        color = DETECTION_COLORS.get(d['class'], (255, 255, 255))
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    marker_y = (bottom_roi_y + int(h * ROI_END_HEIGHT)) // 2
    if yellow_pos:
        cv2.line(display, (yellow_pos, bottom_roi_y), (yellow_pos, int(h * ROI_END_HEIGHT)), (0, 255, 255), 2)
    if round_pos:
        cv2.line(display, (round_pos, bottom_roi_y), (round_pos, int(h * ROI_END_HEIGHT)), (255, 0, 255), 2)
    if target_pos:
        cv2.line(display, (target_pos, bottom_roi_y), (target_pos, int(h * ROI_END_HEIGHT)), (0, 255, 0), 3)

    # Status bar
    cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
    mode_color = (0, 255, 0) if "DRIVING" in mode else (0, 0, 255)
    cv2.putText(display, f'{mode} | T:{throttle:.2f} S:{steering:+.2f} F:{frameCount}', 
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_color, 1)
    
    # LIDAR status
    valid_mask = filter_lidar()
    valid_count = np.sum(valid_mask)
    if valid_count > 0:
        closest = lidar_distances[valid_mask].min()
        cv2.putText(display, f'LIDAR: {valid_count} pts | CLOSEST: {closest:.2f}m', 
                    (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # Big warning
        cv2.rectangle(display, (w//2 - 120, 45), (w//2 + 120, 68), (0, 0, 255), -1)
        cv2.putText(display, f'OBSTACLE: {closest:.2f}m', (w//2 - 80, 62), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(display, f'LIDAR: CLEAR', (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    if stopped_at_red_light:
        cv2.rectangle(display, (w//2 - 120, h - 35), (w//2 + 120, h - 10), (0, 0, 255), -1)
        cv2.putText(display, 'WAITING FOR GREEN', (w//2 - 100, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if emergency_stop:
        cv2.putText(display, 'EMERGENCY STOP!', (w - 150, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return display


# === MAIN LOOP ===
try:
    while True:
        myServer.receive(iterations=2, timeout=timeout)
        lidarRecvFlag, _ = myLidarServer.receive(iterations=1, timeout=timeout)
        
        if lidarRecvFlag:
            process_lidar_data()

        frameCount += 1
        frame = myServer.receiveBuffer.copy()
        all_detections = detect_all_objects(frame)
        seg_mask, yellow_pos, round_pos = segment_lane(frame)

        target_pos = None
        mode = "STOPPED"
        throttle = 0.0
        steering = 0.0

        if emergency_stop:
            mode = "EMERGENCY STOP"
            
        elif not calibrated:
            mode = "CALIBRATING"
            if yellow_pos is not None and round_pos is not None:
                lane_distance = abs(yellow_pos - round_pos) / 2
                calibrated = True
                print(f"\n✅ CALIBRATED!")
                
        else:
            is_red = check_red_light(all_detections)
            is_yellow = check_yellow_light(all_detections)
            is_green = check_green_light(all_detections)
            is_stop_sign = check_stop_sign(all_detections)
            
            if is_stop_sign and not stopped_at_stop_sign:
                stopped_at_stop_sign = True
                stop_sign_start_time = time.time()
            
            if stopped_at_stop_sign:
                if time.time() - stop_sign_start_time < STOP_SIGN_DURATION:
                    mode = "STOP SIGN"
                else:
                    stopped_at_stop_sign = False
                    stop_sign_start_time = None
            
            elif is_red:
                if not stopped_at_red_light:
                    stopped_at_red_light = True
                mode = "RED LIGHT"
            
            elif stopped_at_red_light and is_green:
                stopped_at_red_light = False
            
            elif stopped_at_red_light:
                mode = "WAITING GREEN"
            
            else:
                target_pos, mode = calculate_target(yellow_pos, round_pos, lane_distance)
                if target_pos is not None:
                    if yellow_pos and round_pos:
                        lane_distance = abs(yellow_pos - round_pos) / 2
                    error = target_pos - FRAME_CENTER
                    steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                    throttle = BASE_THROTTLE * (0.9 if is_yellow else 1.0)
                    if is_yellow:
                        mode = "YELLOW SLOW"

        controlData[0] = throttle
        controlData[1] = steering
        myServer.send(controlData.data)

        display = draw_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, all_detections)
        cv2.imshow('Autonomous + LIDAR', display)
        
        if show_lidar and lidar_frame_count > 0:
            cv2.imshow('LIDAR View', draw_lidar_plot())

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            emergency_stop = not emergency_stop
        elif key == ord('v'):
            show_overlay = not show_overlay
        elif key == ord('l'):
            show_lidar = not show_lidar
            if not show_lidar:
                cv2.destroyWindow('LIDAR View')

        if frameCount % 60 == 0:
            valid_count = np.sum(filter_lidar())
            print(f'F:{frameCount} | {mode} | LIDAR:{valid_count} | T:{throttle:.3f} S:{steering:+.3f}')

except KeyboardInterrupt:
    pass

finally:
    controlData[0] = 0.0
    controlData[1] = 0.0
    for _ in range(3):
        myServer.send(controlData.data)
        time.sleep(0.1)
    myServer.terminate()
    myLidarServer.terminate()
    cv2.destroyAllWindows()
    plt.close('all')
    print('Done!')