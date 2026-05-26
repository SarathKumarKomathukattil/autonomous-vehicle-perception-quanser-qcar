'''laptop_server_lane_autonomous_enet_with_data_logging.py

Autonomous Lane Following + UNIFIED Traffic Detection
- USES ENet (FAST & EFFICIENT - 93.73% IoU!)
- WITH IMAGENET NORMALIZATION (CRITICAL FIX!)
- STOPS at RED light
- STRICT: CANNOT MOVE until YOLO detects GREEN
- SLOWS DOWN by 25% when YELLOW light detected
- STOPS at STOP sign for 3 seconds
- Detects: RED, YELLOW, GREEN, STOP, NO_RIGHT_TURN
- WITH COMPREHENSIVE DATA LOGGING (DRIVING FRAMES ONLY)
- WITH CLEAN CSI FEED WINDOW (NO OVERLAYS)
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
import csv
from datetime import datetime
import os

# === PARAMETERS ===
imageWidth = 640
imageHeight = 480
imageChannels = 3
imageBufferSize = imageHeight * imageWidth * imageChannels

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT = 18001

# === MODEL PATHS ===
LANE_SEG_MODEL_PATH = r'C:\Users\kcksa\Documents\ENet_Lane_Segmentation\runs\exp_20251208_124801\weights\best.pt'
UNIFIED_DETECTION_MODEL_PATH = r'C:\Users\kcksa\Documents\Traffic lights and signs\unified_traffic_detection\exp22\weights\best.pt'

# === SEGMENTATION PARAMETERS ===
NUM_CLASSES = 3
SEG_IMAGE_SIZE = (256, 256)
ROI_START_HEIGHT = 0.2
ROI_END_HEIGHT = 0.9
BOTTOM_PORTION = 0.3

# === CONTROL PARAMETERS ===
BASE_THROTTLE = 0.055  # ← CHANGE THIS for different speeds (0.03, 0.06, 0.09, 0.12)
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

# === CLEAN CSI FEED WINDOW ===
SHOW_CLEAN_FEED = True  # Set to False to disable clean feed window

# ========================================
# DATA LOGGING CONFIGURATION
# ========================================

# !!! CHANGE THESE FOR EACH TEST !!!
TEST_METADATA = {
    'model_name': 'ENet',  # ← CHANGE: 'UNet' or 'ENet'
    'test_type': 'moving_car',  # ← CHANGE: 'stationary' or 'moving_car'
    'condition': 'normal_lighting',  # ← CHANGE: 'normal_lighting', 'dim', 'dark', 'bright', '0_occlusion', '25_occlusion', etc.
    'speed_setting': f'{BASE_THROTTLE:.3f}',  # ← AUTOMATICALLY SET from BASE_THROTTLE
    'track_section': 'straight',  # ← CHANGE: 'straight', 'curve', 'roundabout'
}

# Data storage
frame_data = []

# Timing
test_start_time = None
last_log_time = None

# FPS tracking
fps_start_time = None
fps_frame_count = 0
current_fps = 0.0

# Distance tracking
total_distance = 0.0

# Pixel to meter conversion (adjust based on your calibration)
PIXEL_TO_METER = 0.01

# Create output directory if it doesn't exist
OUTPUT_DIR = r'C:\Users\kcksa\Desktop\Testing results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create filenames with metadata
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(OUTPUT_DIR, f"test_data_{TEST_METADATA['model_name']}_{TEST_METADATA['speed_setting']}_{TEST_METADATA['condition']}_{timestamp}.csv")
summary_filename = os.path.join(OUTPUT_DIR, f"summary_{TEST_METADATA['model_name']}_{TEST_METADATA['speed_setting']}_{timestamp}.txt")

print("=" * 60)
print("=== AUTONOMOUS Lane Following + UNIFIED DETECTION ===")
print("=== USING ENet (0.36M params, 93.73% IoU!) ===")
print("=== WITH YELLOW LIGHT SLOW DOWN (25% reduction) ===")
print("=== WITH DATA LOGGING (DRIVING FRAMES ONLY) ===")
print("=== WITH CLEAN CSI FEED WINDOW ===")
print("=" * 60)
print(f"\n📊 DATA LOGGING ENABLED")
print(f"   Model: {TEST_METADATA['model_name']}")
print(f"   Test Type: {TEST_METADATA['test_type']}")
print(f"   Condition: {TEST_METADATA['condition']}")
print(f"   Throttle: {TEST_METADATA['speed_setting']}")
print(f"   Track: {TEST_METADATA['track_section']}")
print(f"   Output Dir: {OUTPUT_DIR}")
print(f"   📌 NOTE: Only DRIVING frames will be saved")
print(f"   📺 Clean CSI Feed: {'ENABLED' if SHOW_CLEAN_FEED else 'DISABLED'}")
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

print(f"\n✅ Server started on port {SERVER_PORT}")
print("⏳ Waiting for QCar connection...")

while not myServer.connected:
    myServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)

print("\n✅ QCar Connected!")

# === LOAD ENet LANE SEGMENTATION MODEL ===
print("\nLoading ENet lane segmentation model...")


class InitialBlock(nn.Module):
    """ENet initial block"""
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels + in_channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        main = self.conv(x)
        side = self.maxpool(x)
        x = torch.cat([main, side], dim=1)
        x = self.bn(x)
        return self.prelu(x)


class BottleneckDownsample(nn.Module):
    """ENet downsampling bottleneck"""
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        
        self.main_max = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=2, 
                              stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        self.prelu_out = nn.PReLU()
    
    def forward(self, x):
        main, max_indices = self.main_max(x)
        
        if main.size(1) != self.bn3.num_features:
            padding = torch.zeros(main.size(0), 
                                self.bn3.num_features - main.size(1),
                                main.size(2), main.size(3))
            if x.is_cuda:
                padding = padding.cuda()
            main = torch.cat([main, padding], dim=1)
        
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.prelu1(ext)
        
        ext = self.conv2(ext)
        ext = self.bn2(ext)
        ext = self.prelu2(ext)
        
        ext = self.conv3(ext)
        ext = self.bn3(ext)
        ext = self.dropout(ext)
        
        out = main + ext
        return self.prelu_out(out), max_indices


class BottleneckRegular(nn.Module):
    """ENet regular bottleneck"""
    def __init__(self, channels, internal_ratio=4, dropout_prob=0.1, dilation=1, asymmetric=False):
        super().__init__()
        internal_channels = channels // internal_ratio
        
        self.conv1 = nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), 
                         padding=(2, 0), bias=False),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), 
                         padding=(0, 2), bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3,
                                  padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        self.prelu_out = nn.PReLU()
    
    def forward(self, x):
        main = x
        
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.prelu1(ext)
        
        ext = self.conv2(ext)
        ext = self.bn2(ext)
        ext = self.prelu2(ext)
        
        ext = self.conv3(ext)
        ext = self.bn3(ext)
        ext = self.dropout(ext)
        
        out = main + ext
        return self.prelu_out(out)


class BottleneckUpsample(nn.Module):
    """ENet upsampling bottleneck"""
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        
        self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_main = nn.BatchNorm2d(out_channels)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        
        self.convt = nn.ConvTranspose2d(internal_channels, internal_channels, 
                                       kernel_size=3, stride=2, padding=1, 
                                       output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        self.prelu_out = nn.PReLU()
    
    def forward(self, x, max_indices):
        main = self.conv_main(x)
        main = self.bn_main(main)
        main = self.unpool(main, max_indices)
        
        ext = self.conv1(x)
        ext = self.bn1(ext)
        ext = self.prelu1(ext)
        
        ext = self.convt(ext)
        ext = self.bn2(ext)
        ext = self.prelu2(ext)
        
        ext = self.conv3(ext)
        ext = self.bn3(ext)
        ext = self.dropout(ext)
        
        out = main + ext
        return self.prelu_out(out)


class ENet(nn.Module):
    """ENet for semantic segmentation"""
    def __init__(self, num_classes):
        super().__init__()
        
        self.initial = InitialBlock(3, 13)
        
        self.downsample1_0 = BottleneckDownsample(16, 64, dropout_prob=0.01)
        self.regular1_1 = BottleneckRegular(64, dropout_prob=0.01)
        self.regular1_2 = BottleneckRegular(64, dropout_prob=0.01)
        self.regular1_3 = BottleneckRegular(64, dropout_prob=0.01)
        self.regular1_4 = BottleneckRegular(64, dropout_prob=0.01)
        
        self.downsample2_0 = BottleneckDownsample(64, 128, dropout_prob=0.1)
        self.regular2_1 = BottleneckRegular(128, dropout_prob=0.1)
        self.dilated2_2 = BottleneckRegular(128, dilation=2, dropout_prob=0.1)
        self.asymmetric2_3 = BottleneckRegular(128, asymmetric=True, dropout_prob=0.1)
        self.dilated2_4 = BottleneckRegular(128, dilation=4, dropout_prob=0.1)
        self.regular2_5 = BottleneckRegular(128, dropout_prob=0.1)
        self.dilated2_6 = BottleneckRegular(128, dilation=8, dropout_prob=0.1)
        self.asymmetric2_7 = BottleneckRegular(128, asymmetric=True, dropout_prob=0.1)
        self.dilated2_8 = BottleneckRegular(128, dilation=16, dropout_prob=0.1)
        
        self.regular3_0 = BottleneckRegular(128, dropout_prob=0.1)
        self.dilated3_1 = BottleneckRegular(128, dilation=2, dropout_prob=0.1)
        self.asymmetric3_2 = BottleneckRegular(128, asymmetric=True, dropout_prob=0.1)
        self.dilated3_3 = BottleneckRegular(128, dilation=4, dropout_prob=0.1)
        self.regular3_4 = BottleneckRegular(128, dropout_prob=0.1)
        self.dilated3_5 = BottleneckRegular(128, dilation=8, dropout_prob=0.1)
        self.asymmetric3_6 = BottleneckRegular(128, asymmetric=True, dropout_prob=0.1)
        self.dilated3_7 = BottleneckRegular(128, dilation=16, dropout_prob=0.1)
        
        self.upsample4_0 = BottleneckUpsample(128, 64, dropout_prob=0.1)
        self.regular4_1 = BottleneckRegular(64, dropout_prob=0.1)
        self.regular4_2 = BottleneckRegular(64, dropout_prob=0.1)
        
        self.upsample5_0 = BottleneckUpsample(64, 16, dropout_prob=0.1)
        self.regular5_1 = BottleneckRegular(16, dropout_prob=0.1)
        
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, 
                                            stride=2, bias=False)
    
    def forward(self, x):
        x = self.initial(x)
        
        x, max_indices1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        
        x, max_indices2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        
        x = self.upsample4_0(x, max_indices2)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        
        x = self.upsample5_0(x, max_indices1)
        x = self.regular5_1(x)
        
        x = self.final_conv(x)
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    lane_model = ENet(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(LANE_SEG_MODEL_PATH, map_location=device, weights_only=False)
    lane_model.load_state_dict(checkpoint['model_state_dict'])
    lane_model.eval()
    print("✅ ENet lane segmentation model loaded!")
    print(f"   Parameters: 0.36M (87x smaller than U-Net!)")
    print(f"   Accuracy: 93.73% IoU")
except Exception as e:
    print(f"❌ Could not load ENet model: {e}")
    import traceback
    traceback.print_exc()
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
    """Run lane segmentation using ENet - WITH IMAGENET NORMALIZATION (CRITICAL!)"""
    global yellow_count, round_count, bottom_roi_y
    
    if lane_model is None:
        return None, None, None

    img_resized = cv2.resize(frame, SEG_IMAGE_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # === IMAGENET NORMALIZATION (CRITICAL FIX!) ===
    # ENet was trained with ImageNet mean/std normalization
    img_normalized = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
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
    cv2.rectangle(status_overlay, (0, 0), (w, 80), (0, 0, 0), -1)
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

    cv2.putText(display, f'Y:{yellow_count}px R:{round_count}px | FPS:{current_fps:.1f}', 
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    if detections:
        det_status = " ".join([f"{d['class']}:{d['conf']:.0%}(H:{d['height']})" for d in detections])
        cv2.putText(display, f'Detections: {det_status}', 
                    (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
    
    # Show data logging status
    cv2.putText(display, f'📊 LOGGING: Dist:{total_distance:.1f}m Time:{time.time()-test_start_time if test_start_time else 0:.1f}s', 
                (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

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


def draw_clean_feed(frame):
    """Draw clean CSI feed - PURE camera feed with NO lane detection or sign detection"""
    clean_display = frame.copy()
    h, w = frame.shape[:2]
    
    # Optional minimal status bar (uncomment if you want ZERO overlays)
    status_overlay = clean_display.copy()
    cv2.rectangle(status_overlay, (0, 0), (w, 25), (0, 0, 0), -1)
    clean_display = cv2.addWeighted(status_overlay, 0.3, clean_display, 0.7, 0)
    
    # Minimal info: just frame count and FPS
    cv2.putText(clean_display, f'CSI Feed | Frame:{frameCount} | FPS:{current_fps:.1f}', 
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return clean_display


# === MAIN LOOP ===
try:
    while True:
        recvFlag, bytesReceived = myServer.receive(iterations=2, timeout=timeout)

        if recvFlag and bytesReceived > 0:
            frameCount += 1
            frame = myServer.receiveBuffer.copy()
            
            # ========================================
            # INITIALIZE TIMING
            # ========================================
            if test_start_time is None:
                test_start_time = time.time()
                fps_start_time = time.time()
                last_log_time = time.time()
                print(f"\n🎬 TEST STARTED - Recording data...")
            
            current_time = time.time()
            elapsed_time = current_time - test_start_time
            
            # Calculate FPS
            fps_frame_count += 1
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (current_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = current_time
            
            # Calculate distance traveled
            dt = current_time - last_log_time
            total_distance += throttle * dt
            last_log_time = current_time

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
                    elapsed_time_sign = time.time() - stop_sign_start_time
                    
                    if elapsed_time_sign < STOP_SIGN_DURATION:
                        mode = "STOP SIGN - STOPPED"
                        status = f"Waiting {STOP_SIGN_DURATION - elapsed_time_sign:.1f}s"
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

            # ========================================
            # LOG ALL DATA FOR THIS FRAME
            # ========================================
            lateral_offset_px = (target_pos - FRAME_CENTER) if target_pos is not None else 0
            lateral_offset_m = lateral_offset_px * PIXEL_TO_METER
            
            frame_record = {
                # Timing
                'frame': frameCount,
                'timestamp': current_time,
                'elapsed_time_s': elapsed_time,
                'fps': current_fps,
                
                # Position & Control
                'distance_traveled_m': total_distance,
                'target_position_px': target_pos if target_pos is not None else -1,
                'lateral_offset_px': lateral_offset_px,
                'lateral_offset_m': lateral_offset_m,
                'throttle': throttle,
                'steering': steering,
                
                # Lane Detection
                'yellow_detected': yellow_pos is not None,
                'yellow_position_px': yellow_pos if yellow_pos is not None else -1,
                'yellow_pixel_count': yellow_count,
                'roundabout_detected': round_pos is not None,
                'roundabout_position_px': round_pos if round_pos is not None else -1,
                'roundabout_pixel_count': round_count,
                'lane_distance_px': lane_distance if lane_distance is not None else -1,
                
                # System State
                'mode': mode,
                'calibrated': calibrated,
                'emergency_stop': emergency_stop,
                'stopped_at_red': stopped_at_red_light,
                'stopped_at_stop_sign': stopped_at_stop_sign,
                
                # Traffic Detection
                'num_detections': len(all_detections),
                'red_light_detected': any(d['class'] == 'red' for d in all_detections),
                'yellow_light_detected': any(d['class'] == 'yellow' for d in all_detections),
                'green_light_detected': any(d['class'] == 'green' for d in all_detections),
                'stop_sign_detected': any(d['class'] == 'stop' for d in all_detections),
                
                # Detection Confidence
                'red_confidence': max([d['conf'] for d in all_detections if d['class'] == 'red'], default=0),
                'yellow_confidence': max([d['conf'] for d in all_detections if d['class'] == 'yellow'], default=0),
                'green_confidence': max([d['conf'] for d in all_detections if d['class'] == 'green'], default=0),
                'stop_confidence': max([d['conf'] for d in all_detections if d['class'] == 'stop'], default=0),
            }
            
            frame_data.append(frame_record)

            controlData[0] = throttle
            controlData[1] = steering
            myServer.send(controlData.data)

            # ========================================
            # DISPLAY WINDOWS
            # ========================================
            
            # Main display with all overlays
            display = draw_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, status, all_detections)
            cv2.imshow('ENet Autonomous Lane Following [DATA LOGGING]', display)

            # Clean CSI feed window (NO overlays)
            if SHOW_CLEAN_FEED:
                clean_feed = draw_clean_feed(frame)
                cv2.imshow('Clean CSI Feed', clean_feed)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print('\n🛑 Quitting and saving data...')
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

            if frameCount % 60 == 0 or stopped_at_red_light or stopped_at_stop_sign:
                det_info = ", ".join([f"{d['class']}({d['conf']:.0%},H:{d['height']})" for d in all_detections]) if all_detections else "None"
                print(f'F:{frameCount} | {mode} | Det:[{det_info}] | T:{throttle:.3f} S:{steering:+.3f} | D:{total_distance:.1f}m')

except KeyboardInterrupt:
    print("\n⚠️  Keyboard interrupt - saving data...")

finally:
    print("\n🛑 Shutting down and saving data...")
    
    # ========================================
    # AUTO-FILTER: KEEP ONLY DRIVING FRAMES
    # ========================================
    
    if frame_data and len(frame_data) > 0:
        print(f"\n🔍 FILTERING TO DRIVING FRAMES ONLY...")
        original_count = len(frame_data)
        
        # Define driving modes (when car is actually moving)
        DRIVING_MODES = ['DRIVING', 'YELLOW ONLY', 'ROUND ONLY', 'YELLOW SLOW (75%)']
        
        # Filter: Keep only frames where throttle > 0 AND in driving mode
        filtered_frames = [
            frame for frame in frame_data 
            if frame['throttle'] > 0 and frame['mode'] in DRIVING_MODES
        ]
        
        if len(filtered_frames) > 0:
            # Get time range of actual driving
            start_time = filtered_frames[0]['elapsed_time_s']
            end_time = filtered_frames[-1]['elapsed_time_s']
            
            # Reset frame numbers and timestamps to start from 0
            for i, frame in enumerate(filtered_frames, start=1):
                frame['frame'] = i
                frame['elapsed_time_s'] = frame['elapsed_time_s'] - start_time
            
            # Replace with filtered data
            frame_data = filtered_frames
            
            print(f"   ✅ Filtered: {original_count} → {len(frame_data)} frames")
            print(f"   ✅ Removed: {original_count - len(frame_data)} non-driving frames ({(original_count - len(frame_data))/original_count*100:.1f}%)")
            print(f"   ✅ Duration: {end_time - start_time:.1f}s of actual driving")
        else:
            print(f"   ⚠️  WARNING: No driving frames found!")
            print(f"   ⚠️  Car may not have moved. Check if calibration completed.")
    
    # ========================================
    # SAVE DATA (CSV + SUMMARY)
    # ========================================
    
    if frame_data and len(frame_data) > 0:
        print(f"\n📊 SAVING DATA...")
        
        # Save frame-by-frame CSV data
        try:
            with open(log_filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=frame_data[0].keys())
                writer.writeheader()
                writer.writerows(frame_data)
            print(f"   ✅ CSV saved: {os.path.basename(log_filename)}")
            print(f"      Frames: {len(frame_data)}")
        except Exception as e:
            print(f"   ❌ Error saving CSV: {e}")
        
        # Calculate and save summary statistics
        try:
            # Filter valid data for statistics
            valid_offsets = [d['lateral_offset_px'] for d in frame_data if d['target_position_px'] != -1]
            valid_yellow = [d['yellow_pixel_count'] for d in frame_data if d['yellow_detected']]
            valid_round = [d['roundabout_pixel_count'] for d in frame_data if d['roundabout_detected']]
            
            summary_stats = {
                'total_frames': len(frame_data),
                'total_time_s': frame_data[-1]['elapsed_time_s'] if frame_data else 0,
                'total_distance_m': frame_data[-1]['distance_traveled_m'] if frame_data else 0,
                'avg_fps': np.mean([d['fps'] for d in frame_data if d['fps'] > 0]) if frame_data else 0,
                
                # Offset statistics
                'mean_offset_px': np.mean(np.abs(valid_offsets)) if valid_offsets else 0,
                'max_offset_px': np.max(np.abs(valid_offsets)) if valid_offsets else 0,
                'std_offset_px': np.std(valid_offsets) if valid_offsets else 0,
                'rms_offset_px': np.sqrt(np.mean(np.array(valid_offsets)**2)) if valid_offsets else 0,
                'mean_offset_m': np.mean(np.abs(valid_offsets)) * PIXEL_TO_METER if valid_offsets else 0,
                'max_offset_m': np.max(np.abs(valid_offsets)) * PIXEL_TO_METER if valid_offsets else 0,
                
                # Lane detection statistics
                'yellow_detection_rate_%': sum(1 for d in frame_data if d['yellow_detected']) / len(frame_data) * 100 if frame_data else 0,
                'round_detection_rate_%': sum(1 for d in frame_data if d['roundabout_detected']) / len(frame_data) * 100 if frame_data else 0,
                'avg_yellow_pixels': np.mean(valid_yellow) if valid_yellow else 0,
                'avg_round_pixels': np.mean(valid_round) if valid_round else 0,
                
                # Control statistics
                'avg_throttle': np.mean([d['throttle'] for d in frame_data]),
                'avg_steering': np.mean([d['steering'] for d in frame_data]),
                'max_steering': np.max(np.abs([d['steering'] for d in frame_data])),
                
                # Success metrics
                'lane_departures': sum(1 for d in frame_data if not d['yellow_detected'] and not d['roundabout_detected']),
            }
            
            # Save summary TXT
            with open(summary_filename, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("TEST SUMMARY (DRIVING FRAMES ONLY)\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("TEST METADATA:\n")
                for key, value in TEST_METADATA.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write(f"\n  test_date: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write(f"  test_time: {datetime.now().strftime('%H:%M:%S')}\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("PERFORMANCE METRICS:\n")
                f.write("=" * 70 + "\n\n")
                for key, value in summary_stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.3f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            print(f"   ✅ Summary saved: {os.path.basename(summary_filename)}")
            
            # Print quick summary to console
            print(f"\n📈 QUICK SUMMARY (DRIVING ONLY):")
            print(f"   Duration: {summary_stats['total_time_s']:.1f}s")
            print(f"   Distance: {summary_stats['total_distance_m']:.1f}m")
            if summary_stats['total_time_s'] > 0:
                print(f"   Avg Speed: {summary_stats['total_distance_m']/summary_stats['total_time_s']:.2f} m/s")
            print(f"   Avg FPS: {summary_stats['avg_fps']:.1f}")
            print(f"   Avg Throttle: {summary_stats['avg_throttle']:.3f}")
            print(f"   Mean Offset: {summary_stats['mean_offset_px']:.1f}px ({summary_stats['mean_offset_m']:.3f}m)")
            print(f"   Max Offset: {summary_stats['max_offset_px']:.1f}px ({summary_stats['max_offset_m']:.3f}m)")
            print(f"   Lane Departures: {summary_stats['lane_departures']}")
            print(f"   Yellow Detection: {summary_stats['yellow_detection_rate_%']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error calculating summary: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  No data collected to save")
    
    # Stop car
    controlData[0] = 0.0
    controlData[1] = 0.0
    if myServer.connected:
        for _ in range(3):
            myServer.send(controlData.data)
            time.sleep(0.1)
    
    myServer.terminate()
    cv2.destroyAllWindows()
    print('\n✅ Done! All data saved to:')
    print(f'   📁 {OUTPUT_DIR}')