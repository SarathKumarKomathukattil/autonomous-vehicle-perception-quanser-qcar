'''laptop_server_lane_autonomous_enet_with_depth.py

Autonomous Lane Following + UNIFIED Traffic Detection + RealSense Depth
- CSI Camera for RGB (lane detection + YOLO)
- RealSense for Depth (obstacle detection)
- Separate windows for RGB and Depth visualization
- Emergency braking on obstacle detection
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
rgbBufferSize = imageHeight * imageWidth * imageChannels

# Depth parameters
depthWidth = 640
depthHeight = 480
depthBufferSize = depthHeight * depthWidth * 2  # uint16

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT_RGB = 18001
SERVER_PORT_DEPTH = 18002

# === OBSTACLE DETECTION PARAMETERS ===
OBSTACLE_DISTANCE_THRESHOLD = 1.2  # meters
OBSTACLE_ROI_WIDTH = 0.6  # Center 60% width
# Match lane following ROI (from 20% to 90% of image height)
OBSTACLE_ROI_START_HEIGHT = 0.3  # Same as ROI_START_HEIGHT for lane
OBSTACLE_ROI_END_HEIGHT = 0.6  # Same as ROI_END_HEIGHT for lane
MIN_OBSTACLE_PIXELS = 100
MAX_DISPLAY_DISTANCE = 2.0  # meters for depth visualization

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
BASE_THROTTLE = 0.06
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

# === YELLOW LIGHT SLOWDOWN PARAMETERS ===
YELLOW_LIGHT_MIN_HEIGHT = 7
YELLOW_LIGHT_MAX_HEIGHT = 11
YELLOW_SLOWDOWN_THROTTLE = 0.05

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

print("=" * 70)
print("=== AUTONOMOUS Lane Following + DEPTH OBSTACLE DETECTION ===")
print("=== CSI Camera (RGB) + RealSense (Depth) ===")
print("=" * 70)

# === CREATE RGB SERVER (CSI Camera) ===
rgbData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

rgbServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT_RGB}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=rgbBufferSize,
    receiveBuffer=rgbData,
    nonBlocking=False
)

# === CREATE DEPTH SERVER (RealSense) ===
depthData = np.zeros((depthHeight, depthWidth), dtype=np.uint16)

depthServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT_DEPTH}',
    agent='S',
    sendBufferSize=0,
    recvBufferSize=depthBufferSize,
    receiveBuffer=depthData,
    nonBlocking=False
)

timeout = Timeout(seconds=2, nanoseconds=0)

print(f"✅ RGB Server started on port {SERVER_PORT_RGB}")
print(f"✅ Depth Server started on port {SERVER_PORT_DEPTH}")
print("⏳ Waiting for QCar connections...")

# Wait for both connections
while not (rgbServer.connected and depthServer.connected):
    if not rgbServer.connected:
        rgbServer.checkConnection(timeout=timeout)
    if not depthServer.connected:
        depthServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)

print("\n✅ Both streams connected!")

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
except Exception as e:
    print(f"❌ Could not load ENet model: {e}")
    lane_model = None

# === LOAD UNIFIED DETECTION MODEL ===
print("\nLoading UNIFIED detection model...")
try:
    unified_model = YOLO(UNIFIED_DETECTION_MODEL_PATH)
    print("✅ Unified detection model loaded!")
except Exception as e:
    print(f"❌ Could not load unified model: {e}")
    unified_model = None

print("\n" + "=" * 70)
print("✅ READY - AUTONOMOUS + DEPTH OBSTACLE DETECTION!")
print(f"   Obstacle threshold: {OBSTACLE_DISTANCE_THRESHOLD}m")
print(f"   Obstacle ROI: {int(OBSTACLE_ROI_WIDTH*100)}% width, {int(OBSTACLE_ROI_START_HEIGHT*100)}%-{int(OBSTACLE_ROI_END_HEIGHT*100)}% height (aligned with lane ROI)")
print("=" * 70)

# === STATE ===
throttle = 0.0
steering = 0.0
frameCount = 0

calibrated = False
lane_distance = None
emergency_stop = False

# Obstacle detection state
obstacle_detected = False
min_obstacle_distance = float('inf')

yellow_count = 0
round_count = 0
bottom_roi_y = 0

all_detections = []
show_overlay = True

stopped_at_red_light = False
stopped_at_stop_sign = False
stop_sign_start_time = None
slowing_for_yellow = False

print("\nCONTROLS:")
print("  Q = Quit")
print("  SPACE = Emergency Stop")
print("  V = Toggle overlay")
print("=" * 70)


def check_obstacle(depth_frame):
    """Check for obstacles in center ROI of depth image - aligned with lane following ROI"""
    global min_obstacle_distance, obstacle_detected
    
    h, w = depth_frame.shape
    
    # Define center ROI - aligned with lane following ROI (20% to 90% height)
    roi_x_start = int(w * (1 - OBSTACLE_ROI_WIDTH) / 2)
    roi_x_end = int(w * (1 + OBSTACLE_ROI_WIDTH) / 2)
    roi_y_start = int(h * OBSTACLE_ROI_START_HEIGHT)  # 20% from top
    roi_y_end = int(h * OBSTACLE_ROI_END_HEIGHT)       # 90% from top
    
    # Extract ROI
    roi_depth = depth_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # Convert from millimeters to meters
    roi_depth_meters = roi_depth.astype(np.float32) / 1000.0
    
    # Filter out invalid readings (0 or very large values)
    valid_depths = roi_depth_meters[(roi_depth_meters > 0.15) & (roi_depth_meters < 1.5)]
    
    if len(valid_depths) < MIN_OBSTACLE_PIXELS:
        obstacle_detected = False
        min_obstacle_distance = float('inf')
        return False, float('inf'), (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    
    # Get minimum distance in ROI
    min_dist = np.min(valid_depths)
    min_obstacle_distance = min_dist
    
    # Check if below threshold
    if min_dist < OBSTACLE_DISTANCE_THRESHOLD:
        obstacle_detected = True
        return True, min_dist, (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    else:
        obstacle_detected = False
        return False, min_dist, (roi_x_start, roi_y_start, roi_x_end, roi_y_end)


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
    """Check if YELLOW light should trigger slowdown"""
    for detection in detections:
        if detection['class'] == 'yellow':
            height = detection['height']
            if YELLOW_LIGHT_MIN_HEIGHT <= height <= YELLOW_LIGHT_MAX_HEIGHT:
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
    """Run lane segmentation using ENet"""
    global yellow_count, round_count, bottom_roi_y
    
    if lane_model is None:
        return None, None, None

    img_resized = cv2.resize(frame, SEG_IMAGE_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
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


def draw_rgb_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, status, detections):
    """Draw RGB visualization with lane detection"""
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
            cv2.putText(display, label, (x1 + 3, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Status overlay
    status_overlay = display.copy()
    cv2.rectangle(status_overlay, (0, 0), (w, 85), (0, 0, 0), -1)
    display = cv2.addWeighted(status_overlay, 0.5, display, 0.5, 0)

    if obstacle_detected:
        mode_color = (0, 0, 255)
    elif mode == "DRIVING":
        mode_color = (0, 255, 0)
    elif "RED LIGHT" in mode or "STOP SIGN" in mode:
        mode_color = (0, 0, 255)
    elif "YELLOW LIGHT" in mode:
        mode_color = (0, 255, 255)
    else:
        mode_color = (0, 200, 255)

    cv2.putText(display, f'{mode} | T:{throttle:.2f} S:{steering:+.2f} | F:{frameCount}', 
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_color, 1)

    cv2.putText(display, f'ENet:93.73%IoU | Y:{yellow_count}px R:{round_count}px', 
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # Obstacle distance display
    obstacle_color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
    cv2.putText(display, f'Obstacle: {min_obstacle_distance:.2f}m', 
                (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, obstacle_color, 1)
    
    if detections:
        det_status = " ".join([f"{d['class']}:{d['height']}px" for d in detections[:3]])
        cv2.putText(display, f'Det: {det_status}', 
                    (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)

    # Traffic light indicators
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

    # Obstacle warning banner
    if obstacle_detected:
        cv2.rectangle(display, (w//2 - 150, h - 40), (w//2 + 150, h - 10), (0, 0, 255), -1)
        cv2.putText(display, f'OBSTACLE! {min_obstacle_distance:.2f}m', (w // 2 - 120, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if stopped_at_red_light:
        cv2.rectangle(display, (w//2 - 150, h - 70), (w//2 + 150, h - 50), (0, 0, 255), -1)
        cv2.putText(display, 'WAITING FOR GREEN', (w // 2 - 120, h - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if stopped_at_stop_sign and stop_sign_start_time is not None:
        elapsed = time.time() - stop_sign_start_time
        remaining = max(0, STOP_SIGN_DURATION - elapsed)
        cv2.rectangle(display, (w//2 - 150, h - 100), (w//2 + 150, h - 80), (0, 0, 255), -1)
        cv2.putText(display, f'STOP SIGN - {remaining:.1f}s', (w // 2 - 100, h - 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return display


def draw_depth_display(depth_frame, obstacle_roi):
    """Draw depth visualization in separate window"""
    h, w = depth_frame.shape
    
    # Convert from millimeters to meters and normalize for display
    depth_meters = depth_frame.astype(np.float32) / 1000.0
    depth_normalized = np.clip(depth_meters / MAX_DISPLAY_DISTANCE, 0, 1)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Draw ROI rectangle
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = obstacle_roi
    color = (0, 0, 255) if obstacle_detected else (0, 255, 0)
    cv2.rectangle(depth_colored, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), color, 3)
    
    # Add title bar
    title_overlay = depth_colored.copy()
    cv2.rectangle(title_overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    depth_colored = cv2.addWeighted(title_overlay, 0.6, depth_colored, 0.4, 0)
    
    # Add distance text
    cv2.putText(depth_colored, f'Min Dist: {min_obstacle_distance:.2f}m', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(depth_colored, f'Threshold: {OBSTACLE_DISTANCE_THRESHOLD}m', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(depth_colored, f'ROI: {int(OBSTACLE_ROI_WIDTH*100)}%W, {int(OBSTACLE_ROI_START_HEIGHT*100)}%-{int(OBSTACLE_ROI_END_HEIGHT*100)}%H', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    if obstacle_detected:
        cv2.rectangle(depth_colored, (w//2 - 150, h - 40), (w//2 + 150, h - 10), (0, 0, 255), -1)
        cv2.putText(depth_colored, 'OBSTACLE DETECTED!', (w//2 - 130, h - 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(depth_colored, 'Path Clear', (w - 150, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Color scale legend
    legend_h = 20
    legend_w = 200
    legend_x = w - legend_w - 20
    legend_y = 20
    
    gradient = np.linspace(0, 255, legend_w, dtype=np.uint8).reshape(1, -1)
    gradient = np.repeat(gradient, legend_h, axis=0)
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    
    depth_colored[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = gradient_colored
    cv2.rectangle(depth_colored, (legend_x, legend_y), (legend_x+legend_w, legend_y+legend_h), (255, 255, 255), 1)
    
    cv2.putText(depth_colored, '0m', (legend_x - 5, legend_y + legend_h + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(depth_colored, f'{MAX_DISPLAY_DISTANCE}m', (legend_x + legend_w - 20, legend_y + legend_h + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return depth_colored


# === MAIN LOOP ===
try:
    while True:
        # Receive RGB frame
        rgbRecvFlag, rgbBytesReceived = rgbServer.receive(iterations=2, timeout=timeout)
        
        # Receive Depth frame
        depthRecvFlag, depthBytesReceived = depthServer.receive(iterations=2, timeout=timeout)

        if rgbRecvFlag and rgbBytesReceived > 0 and depthRecvFlag and depthBytesReceived > 0:
            frameCount += 1
            rgb_frame = rgbServer.receiveBuffer.copy()
            depth_frame = depthServer.receiveBuffer.copy()

            # Check for obstacles
            is_obstacle, obstacle_dist, obstacle_roi = check_obstacle(depth_frame)

            # Detect traffic objects
            if frameCount % CHECK_DETECTION_EVERY_N_FRAMES == 0:
                all_detections = detect_all_objects(rgb_frame)

            # Segment lanes
            seg_mask, yellow_pos, round_pos = segment_lane(rgb_frame)

            target_pos = None
            mode = "STOPPED"
            status = ""
            throttle = 0.0
            steering = 0.0

            if emergency_stop:
                mode = "EMERGENCY STOP"
                throttle = 0.0
                steering = 0.0
                
            elif is_obstacle:
                mode = "OBSTACLE DETECTED - EMERGENCY BRAKE"
                throttle = 0.0
                steering = 0.0
                if frameCount % 30 == 0:
                    print(f"\n⚠️  OBSTACLE at {obstacle_dist:.2f}m - EMERGENCY BRAKE! (Frame {frameCount})")
                
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
                
                if is_stop_sign and not stopped_at_stop_sign:
                    stopped_at_stop_sign = True
                    stop_sign_start_time = time.time()
                    slowing_for_yellow = False
                    print(f"\n🛑 STOP SIGN - STOPPING! (Frame {frameCount})")
                
                if stopped_at_stop_sign:
                    elapsed_time = time.time() - stop_sign_start_time
                    
                    if elapsed_time < STOP_SIGN_DURATION:
                        mode = "STOP SIGN - STOPPED"
                        throttle = 0.0
                        steering = 0.0
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
                
                elif is_red:
                    if not stopped_at_red_light:
                        stopped_at_red_light = True
                        slowing_for_yellow = False
                        print(f"\n🛑 RED LIGHT - STOP! (Frame {frameCount})")
                    
                    mode = "RED LIGHT - STOPPED"
                    throttle = 0.0
                    steering = 0.0
                
                elif stopped_at_red_light and is_green:
                    stopped_at_red_light = False
                    slowing_for_yellow = False
                    print(f"\n✅ GREEN LIGHT - GO! (Frame {frameCount})")
                    
                    target_pos, mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                    
                    if target_pos is not None:
                        if yellow_pos is not None and round_pos is not None:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = BASE_THROTTLE
                
                elif stopped_at_red_light:
                    mode = "WAITING FOR GREEN"
                    throttle = 0.0
                    steering = 0.0
                
                elif is_yellow:
                    if not slowing_for_yellow:
                        slowing_for_yellow = True
                        print(f"\n🟡 YELLOW LIGHT - SLOWING DOWN! (Frame {frameCount})")
                    
                    target_pos, base_mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                    mode = f"{base_mode} - YELLOW SLOWDOWN"
                    
                    if target_pos is not None:
                        if yellow_pos is not None and round_pos is not None:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = YELLOW_SLOWDOWN_THROTTLE
                    else:
                        throttle = 0.0
                        steering = 0.0
                
                else:
                    if slowing_for_yellow:
                        slowing_for_yellow = False
                        print(f"\n✅ YELLOW LIGHT CLEARED - RESUMING! (Frame {frameCount})")
                    
                    target_pos, mode, status = calculate_target_with_offsets(yellow_pos, round_pos, lane_distance)
                    
                    if target_pos is not None:
                        if yellow_pos is not None and round_pos is not None:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = BASE_THROTTLE
                    else:
                        throttle = 0.0
                        steering = 0.0

            # Send control commands
            controlData[0] = throttle
            controlData[1] = steering
            rgbServer.send(controlData.data)

            # Display RGB with lane detection
            rgb_display = draw_rgb_display(rgb_frame, seg_mask, yellow_pos, round_pos, 
                                          target_pos, mode, status, all_detections)
            cv2.imshow('CSI Camera - Lane Following & Traffic Detection', rgb_display)

            # Display Depth in separate window
            depth_display = draw_depth_display(depth_frame, obstacle_roi)
            cv2.imshow('RealSense Depth - Obstacle Detection', depth_display)

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

            if frameCount % 60 == 0:
                det_info = ", ".join([f"{d['class']}({d['conf']:.0%},H:{d['height']})" for d in all_detections]) if all_detections else "None"
                print(f'F:{frameCount} | {mode} | Obs:{obstacle_dist:.2f}m | Det:[{det_info}] | T:{throttle:.3f} S:{steering:+.3f}')

except KeyboardInterrupt:
    print("\n\n⚠️  Keyboard interrupt - stopping...")

except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🛑 Shutting down...")
    controlData[0] = 0.0
    controlData[1] = 0.0
    if rgbServer.connected:
        for _ in range(3):
            rgbServer.send(controlData.data)
            time.sleep(0.1)
    rgbServer.terminate()
    depthServer.terminate()
    cv2.destroyAllWindows()
    print('✅ Done!')