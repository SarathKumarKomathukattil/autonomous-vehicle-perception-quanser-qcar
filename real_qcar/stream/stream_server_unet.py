'''laptop_server_lane_autonomous_with_traffic_light_detection.py

Autonomous Lane Following + UNIFIED Traffic Detection (ONE MODEL)
- SLOWS DOWN by 25% when YELLOW light detected
- STOPS at 10px RED
- STRICT: CANNOT MOVE until YOLO detects GREEN
- STOPS at STOP sign (64px) for 3 seconds
- Detects: RED, YELLOW, GREEN, STOP, NO_RIGHT_TURN
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

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

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
BASE_THROTTLE = 0.0625
YELLOW_THROTTLE_REDUCTION = 0.10  # 25% reduction for yellow light
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

# === TRAFFIC SIGNS HEIGHT PARAMETERS ===
SIGNS_MIN_HEIGHT = 50
SIGNS_MAX_HEIGHT = 100

# === STOP SIGN PARAMETERS ===
STOP_SIGN_MIN_HEIGHT = 40
STOP_SIGN_STOP_HEIGHT = 45
STOP_SIGN_MAX_HEIGHT = 50
STOP_SIGN_DURATION = 3.0

# === OVERLAY SETTINGS ===
OVERLAY_ALPHA = 0.5
YELLOW_LINE_COLOR = (0, 255, 255)
ROUNDABOUT_COLOR = (255, 0, 255)

# Detection colors (BGR)
DETECTION_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'stop': (0, 0, 255),
    'speed_60': (0, 165, 255),
    'no_right_turn': (255, 0, 255)
}

print("=" * 60)
print("=== AUTONOMOUS Lane Following + UNIFIED DETECTION ===")
print("=== WITH YELLOW LIGHT SLOW DOWN (25% reduction) ===")
print("=" * 60)

# === CREATE SERVER ===
imageData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

myServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=imageBufferSize,
    receiveBuffer=imageData,
    nonBlocking=False
)

timeout = Timeout(seconds=2, nanoseconds=0)

print(f"✅ Server started on port {SERVER_PORT}")
print("⏳ Waiting for QCar connection...")

while not myServer.connected:
    myServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)

print("\n✅ QCar Connected!")

# === LOAD LANE SEGMENTATION MODEL ===
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

# === LOAD UNIFIED DETECTION MODEL ===
print("\nLoading UNIFIED detection model...")
try:
    unified_model = YOLO(UNIFIED_DETECTION_MODEL_PATH)
    print("✅ Unified detection model loaded!")
    print("   Detects: RED, YELLOW, GREEN, STOP, NO_RIGHT_TURN")
except Exception as e:
    print(f"❌ Could not load unified model: {e}")
    unified_model = None

print("\n" + "=" * 60)
print("✅ READY - AUTONOMOUS + UNIFIED DETECTION!")
print(f"   Traffic Lights: {MIN_HEIGHT_THRESHOLD}-{MAX_HEIGHT_THRESHOLD}px")
print(f"   Traffic Signs: {SIGNS_MIN_HEIGHT}-{SIGNS_MAX_HEIGHT}px")
print(f"   YOLO ROI: Top {int(YOLO_ROI_HEIGHT*100)}% of frame")
print(f"   Detection: EVERY FRAME")
print(f"   🟡 YELLOW → SLOW DOWN 25% (easier braking)")
print(f"   🔴 RED at {STOP_HEIGHT_THRESHOLD}px → STOP")
print(f"   ⛔ STRICT: NO MOVE until GREEN detected")
print(f"   🟢 GREEN → GO IMMEDIATELY")
print(f"   🛑 STOP SIGN at {STOP_SIGN_STOP_HEIGHT}px → STOP for {STOP_SIGN_DURATION}s")
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

print("\nCONTROLS:")
print("  Q = Quit")
print("  SPACE = Emergency Stop")
print("  V = Toggle overlay")
print("=" * 60)


def detect_all_objects(frame):
    """Detect ALL traffic objects using UNIFIED model"""
    if unified_model is None:
        return []
    
    yolo_roi_end = int(imageHeight * YOLO_ROI_HEIGHT)
    roi_frame = frame[0:yolo_roi_end, :]
    
    results = unified_model(roi_frame, conf=DETECTION_CONF, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = unified_model.names[cls]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox_height = int(y2 - y1)
            bbox_width = int(x2 - x1)
            
            detections.append({
                'class': class_name,
                'conf': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'height': bbox_height,
                'width': bbox_width
            })
    
    return detections


def check_red_light(detections):
    """Check if RED light should trigger STOP"""
    global stopped_at_red_light
    
    for detection in detections:
        if detection['class'] == 'red':
            height = detection['height']
            
            if MIN_HEIGHT_THRESHOLD <= height <= MAX_HEIGHT_THRESHOLD:
                if not stopped_at_red_light:
                    if height >= STOP_HEIGHT_THRESHOLD:
                        return True
                    else:
                        return False
                else:
                    return True
    
    return False


def check_yellow_light(detections):
    """Check if YELLOW light is detected (for slow down)"""
    for detection in detections:
        if detection['class'] == 'yellow':
            height = detection['height']
            if MIN_HEIGHT_THRESHOLD <= height <= MAX_HEIGHT_THRESHOLD:
                return True
    return False


def check_green_light(detections):
    """Check if GREEN light is detected"""
    for detection in detections:
        if detection['class'] == 'green':
            height = detection['height']
            if MIN_HEIGHT_THRESHOLD <= height <= MAX_HEIGHT_THRESHOLD:
                return True
    return False


def check_stop_sign(detections):
    """Check if STOP sign should trigger stop"""
    global stopped_at_stop_sign
    
    for detection in detections:
        if detection['class'] == 'stop':
            height = detection['height']
            
            if STOP_SIGN_MIN_HEIGHT <= height <= STOP_SIGN_MAX_HEIGHT:
                if not stopped_at_stop_sign:
                    if height >= STOP_SIGN_STOP_HEIGHT:
                        return True
                    else:
                        return False
                else:
                    return True
    
    return False


def segment_lane(frame):
    """Run lane segmentation"""
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
    
    roi_height = roi_end - roi_start
    bottom_start = roi_start + int(roi_height * BOTTOM_PORTION)
    bottom_roi_y = bottom_start
    
    bottom_roi_mask = seg_mask[bottom_start:roi_end, :]

    yellow_pixels = np.where(bottom_roi_mask == 1)
    yellow_count = len(yellow_pixels[1])
    yellow_pos = None
    if yellow_count > MIN_PIXELS:
        yellow_pos = int(np.mean(yellow_pixels[1]))

    round_pixels = np.where(bottom_roi_mask == 2)
    round_count = len(round_pixels[1])
    round_pos = None
    if round_count > MIN_PIXELS:
        round_pos = int(np.mean(round_pixels[1]))

    return seg_mask, yellow_pos, round_pos


def calculate_target_with_offsets(yellow_pos, round_pos, lane_distance):
    """Calculate target position"""
    target_pos = None
    mode = "STOPPED"
    status = ""
    
    if yellow_pos is not None and round_pos is not None:
        adjusted_round = round_pos + ROUNDABOUT_OFFSET
        adjusted_yellow = yellow_pos + YELLOW_OFFSET
        target_pos = (adjusted_round + adjusted_yellow) // 2
        mode = "DRIVING"
        status = "Both visible"
        
    elif yellow_pos is not None and lane_distance is not None:
        target_pos = int(yellow_pos - lane_distance + YELLOW_OFFSET)
        mode = "YELLOW ONLY"
        status = "Following yellow"
        
    elif round_pos is not None and lane_distance is not None:
        target_pos = int(round_pos + lane_distance + ROUNDABOUT_OFFSET)
        mode = "ROUND ONLY"
        status = "Following roundabout"
        
    else:
        mode = "NO LANE - STOPPED"
        status = "Lost track!"
        target_pos = None
    
    return target_pos, mode, status


def draw_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, status, detections):
    """Draw visualization"""
    display = frame.copy()
    h, w = frame.shape[:2]
    
    roi_start_y = int(h * ROI_START_HEIGHT)
    roi_end_y = int(h * ROI_END_HEIGHT)
    yolo_roi_end_y = int(h * YOLO_ROI_HEIGHT)

    if seg_mask is not None and show_overlay:
        overlay = display.copy()
        
        yellow_mask = (seg_mask == 1)
        overlay[yellow_mask] = YELLOW_LINE_COLOR
        
        round_mask = (seg_mask == 2)
        overlay[round_mask] = ROUNDABOUT_COLOR
        
        display = cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1 - OVERLAY_ALPHA, 0)
        
        yellow_contours, _ = cv2.findContours(yellow_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, yellow_contours, -1, (0, 200, 200), 2)
        
        round_contours, _ = cv2.findContours(round_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, round_contours, -1, (200, 0, 200), 2)

    if detections:
        for detection in detections:
            det_class = detection['class']
            det_conf = detection['conf']
            x1, y1, x2, y2 = detection['bbox']
            det_height = detection['height']
            
            color = DETECTION_COLORS.get(det_class, (255, 255, 255))
            
            thickness = 3 if det_class in ['red', 'yellow', 'green'] else 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            label = f'{det_class.upper()} {det_conf:.2f} H:{det_height}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            cv2.rectangle(display, 
                         (x1, y1 - label_size[1] - 6), 
                         (x1 + label_size[0] + 6, y1), 
                         color, -1)
            
            cv2.putText(display, label, (x1 + 3, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.line(display, (0, yolo_roi_end_y), (w, yolo_roi_end_y), (255, 255, 255), 2)
    cv2.putText(display, 'YOLO ROI END', (w - 120, yolo_roi_end_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.rectangle(display, (0, roi_start_y), (w, roi_end_y), (100, 100, 100), 1)
    cv2.line(display, (0, bottom_roi_y), (w, bottom_roi_y), (0, 255, 0), 1)

    marker_y = (bottom_roi_y + roi_end_y) // 2

    if yellow_pos is not None:
        cv2.line(display, (yellow_pos, bottom_roi_y), (yellow_pos, roi_end_y), (0, 255, 255), 2)
        cv2.circle(display, (yellow_pos, marker_y), 6, (0, 255, 255), -1)

    if round_pos is not None:
        cv2.line(display, (round_pos, bottom_roi_y), (round_pos, roi_end_y), (255, 0, 255), 2)
        cv2.circle(display, (round_pos, marker_y), 6, (255, 0, 255), -1)

    if target_pos is not None:
        cv2.line(display, (target_pos, bottom_roi_y), (target_pos, roi_end_y), (0, 255, 0), 3)
        cv2.circle(display, (target_pos, marker_y), 8, (0, 255, 0), -1)

    cv2.line(display, (FRAME_CENTER, bottom_roi_y), (FRAME_CENTER, roi_end_y), (255, 255, 255), 1)

    status_overlay = display.copy()
    cv2.rectangle(status_overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    display = cv2.addWeighted(status_overlay, 0.5, display, 0.5, 0)

    if mode == "DRIVING" or mode == "YELLOW SLOW (75%)":
        mode_color = (0, 255, 0)
    elif mode == "CALIBRATING":
        mode_color = (0, 255, 255)
    elif "RED LIGHT" in mode or "WAITING FOR GREEN" in mode or "STOP SIGN" in mode:
        mode_color = (0, 0, 255)
    elif "ONLY" in mode:
        mode_color = (0, 200, 255)
    else:
        mode_color = (0, 0, 255)

    cv2.putText(display, f'{mode} | T:{throttle:.2f} S:{steering:+.2f} F:{frameCount}', 
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_color, 1)

    cv2.putText(display, f'Y:{yellow_count}px R:{round_count}px', 
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    if detections:
        det_status = " ".join([f"{d['class']}:{d['conf']:.0%}(H:{d['height']})" for d in detections])
        cv2.putText(display, f'Detections: {det_status}', 
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)

    lights = [d for d in detections if d['class'] in ['red', 'yellow', 'green']]
    if lights:
        for i, detection in enumerate(lights):
            det_class = detection['class']
            color = DETECTION_COLORS.get(det_class, (255, 255, 255))
            
            circle_x = w - 30 - (i * 50)
            circle_y = 25
            
            cv2.circle(display, (circle_x, circle_y), 18, color, -1)
            cv2.circle(display, (circle_x, circle_y), 18, (255, 255, 255), 2)
            
            cv2.putText(display, det_class[0].upper(), (circle_x - 6, circle_y + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if stopped_at_red_light:
        cv2.rectangle(display, (w//2 - 150, h - 40), (w//2 + 150, h - 10), (0, 0, 255), -1)
        cv2.putText(display, 'WAITING FOR GREEN LIGHT', (w // 2 - 130, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if stopped_at_stop_sign and stop_sign_start_time is not None:
        elapsed = time.time() - stop_sign_start_time
        remaining = max(0, STOP_SIGN_DURATION - elapsed)
        cv2.rectangle(display, (w//2 - 150, h - 70), (w//2 + 150, h - 50), (0, 0, 255), -1)
        cv2.putText(display, f'STOP SIGN - {remaining:.1f}s', (w // 2 - 100, h - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if emergency_stop:
        cv2.putText(display, 'EMERGENCY STOP!', (w - 150, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return display


# === MAIN LOOP ===
try:
    while True:
        recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)

        if recvFlag and bytesReceived > 0:
            frameCount += 1
            frame = myServer.receiveBuffer.copy()

            if frameCount % CHECK_DETECTION_EVERY_N_FRAMES == 0:
                all_detections = detect_all_objects(frame)

            seg_mask, yellow_pos, round_pos = segment_lane(frame)

            target_pos = None
            mode = "STOPPED"
            status = ""
            throttle = 0.0
            steering = 0.0

            if emergency_stop:
                mode = "EMERGENCY STOP"
                status = "Press SPACE to resume"
                stopped_at_red_light = False
                stopped_at_stop_sign = False
                stop_sign_start_time = None
                throttle = 0.0
                steering = 0.0
                
            elif not calibrated:
                mode = "CALIBRATING"
                throttle = 0.0
                steering = 0.0
                if yellow_pos is not None and round_pos is not None:
                    lane_distance = abs(yellow_pos - round_pos) / 2
                    calibrated = True
                    print(f"\n✅ CALIBRATED! Lane distance: {lane_distance:.0f}px")
                    
            else:
                is_red = check_red_light(all_detections)
                is_yellow = check_yellow_light(all_detections)
                is_green = check_green_light(all_detections)
                is_stop_sign = check_stop_sign(all_detections)
                
                # PRIORITY 1: STOP SIGN
                if is_stop_sign and not stopped_at_stop_sign:
                    stopped_at_stop_sign = True
                    stop_sign_start_time = time.time()
                    print(f"\n🛑 STOP SIGN - STOPPING for {STOP_SIGN_DURATION}s! (Frame {frameCount})")
                
                if stopped_at_stop_sign:
                    elapsed_time = time.time() - stop_sign_start_time
                    
                    if elapsed_time < STOP_SIGN_DURATION:
                        mode = "STOP SIGN - STOPPED"
                        status = f"Waiting {STOP_SIGN_DURATION - elapsed_time:.1f}s"
                        throttle = 0.0
                        steering = 0.0
                        target_pos = None
                    else:
                        print(f"\n✅ STOP SIGN COMPLETE - RESUMING! (Frame {frameCount})")
                        stopped_at_stop_sign = False
                        stop_sign_start_time = None
                        
                        target_pos, mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                        
                        if target_pos is not None:
                            if yellow_pos is not None and round_pos is not None:
                                lane_distance = abs(yellow_pos - round_pos) / 2
                            
                            error = target_pos - FRAME_CENTER
                            steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                            throttle = BASE_THROTTLE
                
                # PRIORITY 2: RED LIGHT
                elif is_red:
                    if not stopped_at_red_light:
                        stopped_at_red_light = True
                        print(f"\n🛑 RED LIGHT - STOP! (Frame {frameCount})")
                    
                    mode = "RED LIGHT - STOPPED"
                    status = "Waiting for GREEN"
                    throttle = 0.0
                    steering = 0.0
                    target_pos = None
                
                # STOPPED + GREEN DETECTED → GO!
                elif stopped_at_red_light and is_green:
                    stopped_at_red_light = False
                    print(f"\n✅ GREEN LIGHT - GO! (Frame {frameCount})")
                    
                    target_pos, mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                    
                    if target_pos is not None:
                        if yellow_pos is not None and round_pos is not None:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = BASE_THROTTLE
                
                # STOPPED BUT NO GREEN YET → STAY STOPPED
                elif stopped_at_red_light:
                    mode = "WAITING FOR GREEN"
                    status = "Cannot move without GREEN"
                    throttle = 0.0
                    steering = 0.0
                    target_pos = None
                
                # NORMAL DRIVING (WITH YELLOW LIGHT SLOW DOWN)
                else:
                    target_pos, mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                    
                    if target_pos is not None:
                        if yellow_pos is not None and round_pos is not None:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        
                        # === YELLOW LIGHT SLOW DOWN (25% reduction) ===
                        if is_yellow:
                            throttle = BASE_THROTTLE * (1 - YELLOW_THROTTLE_REDUCTION)
                            mode = "YELLOW SLOW (75%)"
                            if frameCount % 60 == 0:
                                print(f"\n🟡 YELLOW LIGHT - SLOWING DOWN 25%! (Frame {frameCount})")
                        else:
                            throttle = BASE_THROTTLE
                    else:
                        throttle = 0.0
                        steering = 0.0

            controlData[0] = throttle
            controlData[1] = steering
            myServer.send(controlData.data)

            display = draw_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, status, all_detections)
            cv2.imshow('Autonomous Lane Following + Unified Detection', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\nQuitting...')
                break
            elif key == ord(' '):
                emergency_stop = not emergency_stop
                if emergency_stop:
                    print('\n🛑 EMERGENCY STOP!')
                else:
                    print('\n✅ Resumed')
            elif key == ord('v'):
                show_overlay = not show_overlay
                print(f"Overlay: {'ON' if show_overlay else 'OFF'}")

            # === FIXED: Removed is_yellow from condition ===
            if frameCount % 60 == 0 or stopped_at_red_light or stopped_at_stop_sign:
                det_info = ", ".join([f"{d['class']}({d['conf']:.0%},H:{d['height']})" for d in all_detections]) if all_detections else "None"
                print(f'F:{frameCount} | {mode} | Det:[{det_info}] | T:{throttle:.3f} S:{steering:+.3f}')

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    controlData[0] = 0.0
    controlData[1] = 0.0
    if myServer.connected:
        myServer.send(controlData.data)
        time.sleep(0.1)
        myServer.send(controlData.data)
        time.sleep(0.1)
        myServer.send(controlData.data)
    myServer.terminate()
    cv2.destroyAllWindows()
    print('Done!')