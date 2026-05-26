'''laptop_server_lane_autonomous_enet_with_fusion.py

Autonomous Lane Following + UNIFIED Traffic Detection + RealSense Depth + LIDAR FUSION
- CSI Camera for RGB (lane detection + YOLO)
- RealSense for Depth (obstacle detection)
- LIDAR for 360° obstacle detection
- STRICT Sensor Fusion: BOTH sensors must agree to brake
- OPTIMIZED: Reduced rates for depth and LIDAR to prevent interference
- OBSTACLE STOP: Car stops and stays stopped when both sensors detect obstacle
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

# LIDAR parameters
numMeasurements = 360  # MUST MATCH CLIENT!
lidarBufferSize = numMeasurements * 4 * 2
lidarData = np.zeros(numMeasurements * 2, dtype=np.float32)

controlBufferSize = 8
controlData = np.zeros(2, dtype=np.float32)

SERVER_PORT_RGB = 18001
SERVER_PORT_DEPTH = 18002
SERVER_PORT_LIDAR = 18003

# === OBSTACLE DETECTION PARAMETERS ===
# RealSense parameters
DEPTH_OBSTACLE_THRESHOLD = 1.2  # meters
DEPTH_ROI_WIDTH = 0.6  # Center 60% width
DEPTH_ROI_START_HEIGHT = 0.3
DEPTH_ROI_END_HEIGHT = 0.6
MIN_OBSTACLE_PIXELS = 100
MIN_VALID_DEPTH = 0.5  # Ignore depth readings below this (car body/ground)
MAX_VALID_DEPTH = 3.5  # Ignore depth readings above this (noise)

# LIDAR parameters
LIDAR_OBSTACLE_THRESHOLD = 0.8  # meters
LIDAR_MIN_DISTANCE = 0.1
LIDAR_OBSTACLE_MIN_POINTS = 20  # Strong detection (filters signs/lights ≤14 points)
LIDAR_OBSTACLE_MIN_POINTS_PERSISTENCE = 5  # Weak detection for already-tracked obstacles
LIDAR_PERSISTENCE_FRAMES = 10  # Frames to maintain weak tracking after strong detection

# Display parameters
MAX_DISPLAY_DISTANCE = 2.0  # meters for depth visualization
LIDAR_DISPLAY_SIZE = 500

# === MODEL PATHS ===
LANE_SEG_MODEL_PATH = r'C:\Users\kcksa\Documents\ENet_Lane_Segmentation\runs\exp_20251208_124801\weights\best.pt'
UNIFIED_DETECTION_MODEL_PATH = r'C:\Users\kcksa\Documents\Traffic lights and signs\unified_traffic_detection\exp22\weights\best.pt'

# === SEGMENTATION PARAMETERS ===
NUM_CLASSES = 3
SEG_IMAGE_SIZE = (224,224)
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

DETECTION_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'stop': (0, 0, 255),
    'speed_60': (0, 165, 255),
    'no_right_turn': (255, 0, 255)
}

print("=" * 70)
print("=== AUTONOMOUS + STRICT SENSOR FUSION (BOTH MUST AGREE) ===")
print("=== OPTIMIZED: Depth @10Hz, LIDAR @20Hz, Lane @30Hz ===")
print("=== OBSTACLE STOP: Car stops when both sensors detect obstacle ===")
print("=== LIDAR PERSISTENCE: Smart tracking filters signs/lights ===")
print("=" * 70)

# === CREATE SERVERS ===
rgbData = np.zeros((imageHeight, imageWidth, imageChannels), dtype=np.uint8)

rgbServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT_RGB}',
    agent='S',
    sendBufferSize=controlBufferSize,
    recvBufferSize=rgbBufferSize,
    receiveBuffer=rgbData,
    nonBlocking=False
)

depthData = np.zeros((depthHeight, depthWidth), dtype=np.uint16)

depthServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT_DEPTH}',
    agent='S',
    sendBufferSize=0,
    recvBufferSize=depthBufferSize,
    receiveBuffer=depthData,
    nonBlocking=False
)

lidarServer = BasicStream(
    f'tcpip://0.0.0.0:{SERVER_PORT_LIDAR}',
    agent='S',
    sendBufferSize=0,
    recvBufferSize=lidarBufferSize,
    receiveBuffer=lidarData,
    nonBlocking=False
)

timeout = Timeout(seconds=2, nanoseconds=0)
lidar_timeout = Timeout(seconds=0, nanoseconds=10000000)  # 10ms - NON-BLOCKING!
depth_timeout = Timeout(seconds=0, nanoseconds=50000000)  # 50ms - NON-BLOCKING!

print(f"✅ RGB Server started on port {SERVER_PORT_RGB}")
print(f"✅ Depth Server started on port {SERVER_PORT_DEPTH}")
print(f"✅ LIDAR Server started on port {SERVER_PORT_LIDAR}")
print("⏳ Waiting for QCar connections...")

# Wait for RGB (mandatory)
while not rgbServer.connected:
    rgbServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)
print("\n✅ RGB Connected!")

# Wait for Depth (mandatory)
while not depthServer.connected:
    depthServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)
print("✅ Depth Connected!")

# Wait for LIDAR (mandatory)
while not lidarServer.connected:
    lidarServer.checkConnection(timeout=timeout)
    print(".", end="", flush=True)
    time.sleep(0.1)
print("✅ LIDAR Connected!")

print("\n✅ All streams connected!")

# === LIDAR DATA STORAGE ===
lidar_angles = np.zeros(numMeasurements, dtype=np.float32)
lidar_distances = np.zeros(numMeasurements, dtype=np.float32)
lidar_frame_count = 0

# === LOAD ENet MODEL ===
print("\nLoading ENet lane segmentation model...")


class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
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
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.main_max = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.prelu_out = nn.PReLU()
    
    def forward(self, x):
        main, max_indices = self.main_max(x)
        if main.size(1) != self.bn3.num_features:
            padding = torch.zeros(main.size(0), self.bn3.num_features - main.size(1), main.size(2), main.size(3))
            if x.is_cuda:
                padding = padding.cuda()
            main = torch.cat([main, padding], dim=1)
        ext = self.prelu1(self.bn1(self.conv1(x)))
        ext = self.prelu2(self.bn2(self.conv2(ext)))
        ext = self.dropout(self.bn3(self.conv3(ext)))
        return self.prelu_out(main + ext), max_indices


class BottleneckRegular(nn.Module):
    def __init__(self, channels, internal_ratio=4, dropout_prob=0.1, dilation=1, asymmetric=False):
        super().__init__()
        internal_channels = channels // internal_ratio
        self.conv1 = nn.Conv2d(channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(internal_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.prelu_out = nn.PReLU()
    
    def forward(self, x):
        ext = self.prelu1(self.bn1(self.conv1(x)))
        ext = self.prelu2(self.bn2(self.conv2(ext)))
        ext = self.dropout(self.bn3(self.conv3(ext)))
        return self.prelu_out(x + ext)


class BottleneckUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0.1):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_main = nn.BatchNorm2d(out_channels)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        self.convt = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.prelu_out = nn.PReLU()
    
    def forward(self, x, max_indices):
        main = self.unpool(self.bn_main(self.conv_main(x)), max_indices)
        ext = self.prelu1(self.bn1(self.conv1(x)))
        ext = self.prelu2(self.bn2(self.convt(ext)))
        ext = self.dropout(self.bn3(self.conv3(ext)))
        return self.prelu_out(main + ext)


class ENet(nn.Module):
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
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=False)
    
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
        return self.final_conv(x)


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

print("\nLoading UNIFIED detection model...")
try:
    unified_model = YOLO(UNIFIED_DETECTION_MODEL_PATH)
    print("✅ Unified detection model loaded!")
except Exception as e:
    print(f"❌ Could not load unified model: {e}")
    unified_model = None

print("\n" + "=" * 70)
print("✅ READY - STRICT SENSOR FUSION (BOTH MUST AGREE)!")
print(f"   Depth: {DEPTH_OBSTACLE_THRESHOLD}m threshold, valid range: {MIN_VALID_DEPTH}-{MAX_VALID_DEPTH}m")
print(f"   LIDAR: {LIDAR_MIN_DISTANCE}m to {LIDAR_OBSTACLE_THRESHOLD}m range")
print(f"   LIDAR Strong: ≥{LIDAR_OBSTACLE_MIN_POINTS} points (filters signs/lights)")
print(f"   LIDAR Persistence: ≥{LIDAR_OBSTACLE_MIN_POINTS_PERSISTENCE} points for {LIDAR_PERSISTENCE_FRAMES} frames")
print(f"   FUSION RULE: BOTH Depth AND LIDAR must detect → STOP IMMEDIATELY")
print("=" * 70)

# === STATE ===
throttle = 0.0
steering = 0.0
frameCount = 0
depthFrameCount = 0

calibrated = False
lane_distance = None
emergency_stop = False

# Obstacle detection state
depth_obstacle_detected = False
lidar_obstacle_detected = False
fusion_obstacle_detected = False
min_depth_distance = float('inf')
min_lidar_distance = float('inf')
lidar_strong_detection_frame = -999  # Track last strong LIDAR detection

yellow_count = 0
round_count = 0
bottom_roi_y = 0

all_detections = []
show_overlay = True
show_depth = True  # Start with depth OFF for performance
show_lidar = True  # Start with LIDAR OFF for performance

stopped_at_red_light = False
stopped_at_stop_sign = False
stop_sign_start_time = None
slowing_for_yellow = False
stopped_for_obstacle = False

print("\nCONTROLS: Q=Quit, SPACE=Stop, V=Overlay, D=Depth, L=LIDAR")
print("TIP: Keep D and L OFF for best performance (30 FPS)")
print("=" * 70)


def process_lidar_data():
    """Process LIDAR data"""
    global lidar_angles, lidar_distances, lidar_frame_count
    try:
        recv_buffer = lidarServer.receiveBuffer
        for i in range(numMeasurements):
            lidar_angles[i] = recv_buffer[i * 2]
            lidar_distances[i] = recv_buffer[i * 2 + 1]
        lidar_frame_count += 1
        if lidar_frame_count == 1:
            print(f"\n📡 LIDAR: {np.count_nonzero(lidar_distances)}/{numMeasurements} points")
    except Exception as e:
        print(f"⚠ LIDAR error: {e}")


def check_lidar_obstacle():
    """
    Check LIDAR for obstacles with smart persistence
    - Strong detection: ≥15 points (filters signs/lights which give ≤14)
    - Persistence mode: Accept ≥5 points if recently had strong detection
    - Prevents losing real obstacles when point count temporarily drops
    """
    global min_lidar_distance, lidar_obstacle_detected, lidar_strong_detection_frame
    
    valid_mask = (lidar_distances > LIDAR_MIN_DISTANCE) & (lidar_distances < LIDAR_OBSTACLE_THRESHOLD)
    valid_distances = lidar_distances[valid_mask]
    num_points = len(valid_distances)
    
    if num_points == 0:
        lidar_obstacle_detected = False
        min_lidar_distance = float('inf')
        return False, float('inf')
    
    min_dist = np.min(valid_distances)
    
    # STRONG DETECTION: ≥15 points → Real obstacle (not sign/light)
    if num_points >= LIDAR_OBSTACLE_MIN_POINTS:
        lidar_obstacle_detected = True
        min_lidar_distance = min_dist
        lidar_strong_detection_frame = lidar_frame_count
        return True, min_dist
    
    # PERSISTENCE MODE: Recently had strong detection
    elif (lidar_frame_count - lidar_strong_detection_frame) < LIDAR_PERSISTENCE_FRAMES:
        if num_points >= LIDAR_OBSTACLE_MIN_POINTS_PERSISTENCE:
            lidar_obstacle_detected = True
            min_lidar_distance = min_dist
            return True, min_dist
        else:
            lidar_obstacle_detected = False
            min_lidar_distance = float('inf')
            return False, float('inf')
    
    # REJECTION: Low points, no recent strong detection → Sign/light/noise
    else:
        lidar_obstacle_detected = False
        min_lidar_distance = float('inf')
        return False, float('inf')


def check_depth_obstacle(depth_frame):
    """Check RealSense depth for obstacles - with better filtering"""
    global min_depth_distance, depth_obstacle_detected
    
    h, w = depth_frame.shape
    
    roi_x_start = int(w * (1 - DEPTH_ROI_WIDTH) / 2)
    roi_x_end = int(w * (1 + DEPTH_ROI_WIDTH) / 2)
    roi_y_start = int(h * DEPTH_ROI_START_HEIGHT)
    roi_y_end = int(h * DEPTH_ROI_END_HEIGHT)
    
    roi_depth = depth_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    roi_depth_meters = roi_depth.astype(np.float32) / 1000.0
    
    # Better filtering: ignore very close readings (car body) and very far readings (noise)
    valid_depths = roi_depth_meters[
        (roi_depth_meters > MIN_VALID_DEPTH) & 
        (roi_depth_meters < MAX_VALID_DEPTH)
    ]
    
    if len(valid_depths) < MIN_OBSTACLE_PIXELS:
        depth_obstacle_detected = False
        min_depth_distance = float('inf')
        return False, float('inf'), (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    
    min_dist = np.min(valid_depths)
    min_depth_distance = min_dist
    
    if min_dist < DEPTH_OBSTACLE_THRESHOLD:
        depth_obstacle_detected = True
        return True, min_dist, (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    else:
        depth_obstacle_detected = False
        return False, min_dist, (roi_x_start, roi_y_start, roi_x_end, roi_y_end)


def sensor_fusion_decision():
    """
    STRICT sensor fusion: BOTH sensors MUST agree to brake
    - Depth detects obstacle AND LIDAR detects obstacle → BRAKE
    - Only one sensor detects → IGNORE (likely false positive)
    """
    global fusion_obstacle_detected
    
    if depth_obstacle_detected and lidar_obstacle_detected:
        fusion_obstacle_detected = True
        return True
    else:
        fusion_obstacle_detected = False
        return False


def draw_lidar_cv():
    """Draw LIDAR visualization using OpenCV"""
    size = LIDAR_DISPLAY_SIZE
    center = size // 2
    
    img = np.ones((size, size, 3), dtype=np.uint8) * np.array([45, 45, 35], dtype=np.uint8)
    
    max_pixels = int(center * 0.80)
    circle_color = (80, 80, 70)
    
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        radius = int((r / LIDAR_OBSTACLE_THRESHOLD) * max_pixels)
        cv2.circle(img, (center, center), radius, circle_color, 1, cv2.LINE_AA)
        
        label = f'{r:.1f}m'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        label_x = center - text_size[0] // 2
        label_y = center - radius - 5
        
        cv2.rectangle(img, (label_x - 2, label_y - 12), (label_x + text_size[0] + 2, label_y + 2), 
                     (25, 25, 20), -1)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.35, (180, 180, 160), 1, cv2.LINE_AA)
    
    cv2.line(img, (center, 30), (center, size - 30), (100, 100, 90), 1, cv2.LINE_AA)
    cv2.line(img, (30, center), (size - 30, center), (100, 100, 90), 1, cv2.LINE_AA)
    
    directions = [
        ('FRONT', center - 25, 20),
        ('REAR', center - 20, size - 8),
        ('LEFT', 8, center + 5),
        ('RIGHT', size - 42, center + 5)
    ]
    
    for text, x, y in directions:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(img, (x - 3, y - 13), (x + text_size[0] + 3, y + 3), 
                     (25, 25, 20), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (200, 200, 180), 1, cv2.LINE_AA)
    
    valid_mask = (lidar_distances > LIDAR_MIN_DISTANCE) & (lidar_distances < LIDAR_OBSTACLE_THRESHOLD)
    valid_angles = lidar_angles[valid_mask]
    valid_distances = lidar_distances[valid_mask]
    
    if len(valid_distances) == 0:
        status_text = 'No Obstacles'
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center - text_size[0] // 2
        text_y = center
        
        cv2.rectangle(img, (text_x - 10, text_y - 20), (text_x + text_size[0] + 10, text_y + 8),
                     (30, 80, 30), -1)
        cv2.rectangle(img, (text_x - 10, text_y - 20), (text_x + text_size[0] + 10, text_y + 8),
                     (0, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(img, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (100, 255, 100), 2, cv2.LINE_AA)
        
        cv2.putText(img, f'Frame: {lidar_frame_count}', (10, size - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 140), 1, cv2.LINE_AA)
    else:
        for angle, dist in zip(valid_angles, valid_distances):
            px = int(center + (dist / LIDAR_OBSTACLE_THRESHOLD) * max_pixels * np.sin(angle))
            py = int(center - (dist / LIDAR_OBSTACLE_THRESHOLD) * max_pixels * np.cos(angle))
            
            if dist < 0.2:
                color = (80, 80, 255)
            elif dist < 0.35:
                color = (80, 200, 255)
            elif dist < 0.5:
                color = (80, 255, 255)
            else:
                color = (80, 255, 150)
            
            cv2.circle(img, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 6, color, 1, cv2.LINE_AA)
        
        min_idx = np.argmin(valid_distances)
        min_dist = valid_distances[min_idx]
        min_angle = valid_angles[min_idx]
        
        px = int(center + (min_dist / LIDAR_OBSTACLE_THRESHOLD) * max_pixels * np.sin(min_angle))
        py = int(center - (min_dist / LIDAR_OBSTACLE_THRESHOLD) * max_pixels * np.cos(min_angle))
        
        cv2.line(img, (center, center), (px, py), (100, 100, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 10, (100, 100, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 6, (150, 150, 255), -1, cv2.LINE_AA)
        
        panel_height = 70
        cv2.rectangle(img, (0, 0), (size, panel_height), (30, 30, 25), -1)
        cv2.line(img, (0, panel_height), (size, panel_height), (80, 80, 70), 2)
        
        dist_text = f'{min_dist:.2f}m'
        cv2.putText(img, 'Closest:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (180, 180, 160), 1, cv2.LINE_AA)
        cv2.putText(img, dist_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (150, 220, 255), 2, cv2.LINE_AA)
        
        points_text = f'{len(valid_distances)} pts'
        cv2.putText(img, 'Detections:', (size - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (180, 180, 160), 1, cv2.LINE_AA)
        cv2.putText(img, points_text, (size - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (100, 255, 200), 1, cv2.LINE_AA)
        
        angle_deg = int(np.degrees(min_angle))
        angle_text = f'{angle_deg}°'
        cv2.putText(img, 'Angle:', (size // 2 - 60, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (180, 180, 160), 1, cv2.LINE_AA)
        cv2.putText(img, angle_text, (size // 2 - 60, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 200, 100), 1, cv2.LINE_AA)
    
    return img


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
        if d['class'] == 'yellow' and YELLOW_LIGHT_MIN_HEIGHT <= d['height'] <= YELLOW_LIGHT_MAX_HEIGHT:
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
    """Run ENet lane segmentation"""
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


def draw_rgb_display(frame, seg_mask, yellow_pos, round_pos, target_pos, mode, detections):
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
        label = f'{d["class"].upper()} H:{d["height"]}'
        cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if yellow_pos:
        cv2.line(display, (yellow_pos, bottom_roi_y), (yellow_pos, int(h * ROI_END_HEIGHT)), (0, 255, 255), 2)
    if round_pos:
        cv2.line(display, (round_pos, bottom_roi_y), (round_pos, int(h * ROI_END_HEIGHT)), (255, 0, 255), 2)
    if target_pos:
        cv2.line(display, (target_pos, bottom_roi_y), (target_pos, int(h * ROI_END_HEIGHT)), (0, 255, 0), 3)

    # SEMI-TRANSPARENT STATUS BAR (see-through!)
    status_overlay = display.copy()
    cv2.rectangle(status_overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    display = cv2.addWeighted(status_overlay, 0.6, display, 0.4, 0)  # 60% black, 40% original
    
    mode_color = (0, 255, 0) if "DRIVING" in mode else (0, 255, 255) if "YELLOW" in mode else (0, 0, 255)
    cv2.putText(display, f'{mode} | T:{throttle:.2f} S:{steering:+.2f} F:{frameCount}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_color, 1)
    cv2.putText(display, f'ENet | Y:{yellow_count}px R:{round_count}px', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # Sensor fusion status
    depth_color = (0, 0, 255) if depth_obstacle_detected else (0, 255, 0)
    lidar_color = (0, 0, 255) if lidar_obstacle_detected else (0, 255, 0)
    
    depth_status = f'Depth: {min_depth_distance:.2f}m' if min_depth_distance != float('inf') else 'Depth: CLEAR'
    lidar_status = f'LIDAR: {min_lidar_distance:.2f}m' if min_lidar_distance != float('inf') else 'LIDAR: CLEAR'
    
    cv2.putText(display, depth_status, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, depth_color, 1)
    cv2.putText(display, lidar_status, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, lidar_color, 1)
    
    if depth_obstacle_detected and lidar_obstacle_detected:
        fusion_text = 'FUSION: BOTH AGREE - STOPPED!'
        fusion_color = (0, 0, 255)
    elif depth_obstacle_detected:
        fusion_text = 'FUSION: Depth only (IGNORED)'
        fusion_color = (0, 200, 200)
    elif lidar_obstacle_detected:
        fusion_text = 'FUSION: LIDAR only (IGNORED)'
        fusion_color = (0, 200, 200)
    else:
        fusion_text = 'FUSION: CLEAR'
        fusion_color = (0, 255, 0)
    
    cv2.putText(display, fusion_text, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, fusion_color, 1)

    if fusion_obstacle_detected:
        cv2.rectangle(display, (w//2 - 190, h - 40), (w//2 + 190, h - 10), (0, 0, 255), -1)
        warning_text = f'OBSTACLE! D:{min_depth_distance:.2f}m L:{min_lidar_distance:.2f}m - STOPPED'
        cv2.putText(display, warning_text, (w//2 - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    if stopped_at_red_light:
        cv2.rectangle(display, (w//2 - 120, h - 70), (w//2 + 120, h - 50), (0, 0, 255), -1)
        cv2.putText(display, 'WAITING FOR GREEN', (w//2 - 100, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if slowing_for_yellow:
        cv2.rectangle(display, (w//2 - 120, h - 100), (w//2 + 120, h - 80), (0, 255, 255), -1)
        cv2.putText(display, 'YELLOW - SLOWING', (w//2 - 90, h - 87), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return display


def draw_depth_display(depth_frame, obstacle_roi):
    """Draw depth visualization"""
    h, w = depth_frame.shape
    
    depth_meters = depth_frame.astype(np.float32) / 1000.0
    depth_normalized = np.clip(depth_meters / MAX_DISPLAY_DISTANCE, 0, 1)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = obstacle_roi
    color = (0, 0, 255) if depth_obstacle_detected else (0, 255, 0)
    cv2.rectangle(depth_colored, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), color, 3)
    
    title_overlay = depth_colored.copy()
    cv2.rectangle(title_overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    depth_colored = cv2.addWeighted(title_overlay, 0.6, depth_colored, 0.4, 0)
    
    cv2.putText(depth_colored, f'Min: {min_depth_distance:.2f}m | Threshold: {DEPTH_OBSTACLE_THRESHOLD}m', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(depth_colored, f'Frames: {depthFrameCount} | Valid: {MIN_VALID_DEPTH:.1f}-{MAX_VALID_DEPTH:.1f}m', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    if depth_obstacle_detected:
        cv2.rectangle(depth_colored, (w//2 - 100, h - 30), (w//2 + 100, h - 5), (0, 0, 255), -1)
        cv2.putText(depth_colored, 'DETECTED!', (w//2 - 60, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return depth_colored


# === MAIN LOOP ===
try:
    while True:
        # RGB receive (blocking - main control loop)
        rgbRecvFlag, rgbBytesReceived = rgbServer.receive(iterations=2, timeout=timeout)
        
        # Depth receive (NON-BLOCKING)
        depthRecvFlag = False
        try:
            depthRecvFlag, _ = depthServer.receive(iterations=1, timeout=depth_timeout)
            if depthRecvFlag:
                depthFrameCount += 1
        except:
            pass
        
        # LIDAR receive (NON-BLOCKING)
        try:
            lidarRecvFlag, _ = lidarServer.receive(iterations=1, timeout=lidar_timeout)
            if lidarRecvFlag:
                process_lidar_data()
        except:
            pass
        
        if not rgbRecvFlag or rgbBytesReceived == 0:
            continue

        frameCount += 1
        rgb_frame = rgbServer.receiveBuffer.copy()
        depth_frame = depthServer.receiveBuffer.copy()

        # Check obstacles
        is_depth_obstacle, depth_dist, obstacle_roi = check_depth_obstacle(depth_frame)
        is_lidar_obstacle, lidar_dist = check_lidar_obstacle()
        is_obstacle = sensor_fusion_decision()

        all_detections = detect_all_objects(rgb_frame)
        seg_mask, yellow_pos, round_pos = segment_lane(rgb_frame)

        # ═══════════════════════════════════════════════════════════════
        # CONTROL LOGIC - FIXED VERSION
        # ═══════════════════════════════════════════════════════════════
        
        target_pos = None
        mode = "STOPPED"
        throttle = 0.0
        steering = 0.0

        # PRIORITY 1: EMERGENCY STOP
        if emergency_stop:
            mode = "EMERGENCY STOP"
            throttle = 0.0
            steering = 0.0
            stopped_for_obstacle = False
            
        # PRIORITY 2: OBSTACLE DETECTION
        elif is_obstacle:
            if not stopped_for_obstacle:
                stopped_for_obstacle = True
                print(f"\n🛑 OBSTACLE! BOTH SENSORS AGREE - EMERGENCY STOP!")
                print(f"   Depth: {depth_dist:.2f}m | LIDAR: {lidar_dist:.2f}m")
            
            mode = "OBSTACLE DETECTED - STOPPED"
            throttle = 0.0
            steering = 0.0
            target_pos = None
            
        # PRIORITY 3+: NORMAL DRIVING (includes obstacle cleared handling)
        else:
            # Clear obstacle flag if it was set
            if stopped_for_obstacle:
                stopped_for_obstacle = False
                print(f"\n✅ OBSTACLE CLEARED! Resuming... (Frame {frameCount})")
            
            # Calibration
            if not calibrated:
                mode = "CALIBRATING"
                if yellow_pos is not None and round_pos is not None:
                    lane_distance = abs(yellow_pos - round_pos) / 2
                    calibrated = True
                    print(f"\n✅ CALIBRATED! Lane distance: {lane_distance:.0f}px")
                    
            # Autonomous driving
            else:
                is_red = check_red_light(all_detections)
                is_yellow = check_yellow_light(all_detections)
                is_green = check_green_light(all_detections)
                is_stop_sign = check_stop_sign(all_detections)
                
                # Stop sign handling
                if is_stop_sign and not stopped_at_stop_sign:
                    stopped_at_stop_sign = True
                    stop_sign_start_time = time.time()
                    slowing_for_yellow = False
                
                if stopped_at_stop_sign:
                    if time.time() - stop_sign_start_time < STOP_SIGN_DURATION:
                        mode = "STOP SIGN - STOPPED"
                        throttle = 0.0
                        steering = 0.0
                    else:
                        stopped_at_stop_sign = False
                        stop_sign_start_time = None
                        target_pos, mode = calculate_target(yellow_pos, round_pos, lane_distance)
                        if target_pos is not None:
                            if yellow_pos and round_pos:
                                lane_distance = abs(yellow_pos - round_pos) / 2
                            error = target_pos - FRAME_CENTER
                            steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                            throttle = BASE_THROTTLE
                
                # Red light
                elif is_red:
                    if not stopped_at_red_light:
                        stopped_at_red_light = True
                        slowing_for_yellow = False
                    mode = "RED LIGHT - STOPPED"
                    throttle = 0.0
                    steering = 0.0
                
                # Green light (resume from red)
                elif stopped_at_red_light and is_green:
                    stopped_at_red_light = False
                    slowing_for_yellow = False
                    target_pos, mode = calculate_target(yellow_pos, round_pos, lane_distance)
                    if target_pos is not None:
                        if yellow_pos and round_pos:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = BASE_THROTTLE
                
                # Waiting for green
                elif stopped_at_red_light:
                    mode = "WAITING FOR GREEN"
                    throttle = 0.0
                    steering = 0.0
                
                # Yellow light
                elif is_yellow:
                    if not slowing_for_yellow:
                        slowing_for_yellow = True
                    target_pos, base_mode = calculate_target(yellow_pos, round_pos, lane_distance)
                    mode = f"{base_mode} - YELLOW"
                    if target_pos is not None:
                        if yellow_pos and round_pos:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = YELLOW_SLOWDOWN_THROTTLE
                
                # Normal driving
                else:
                    if slowing_for_yellow:
                        slowing_for_yellow = False
                    target_pos, mode = calculate_target(yellow_pos, round_pos, lane_distance)
                    if target_pos is not None:
                        if yellow_pos and round_pos:
                            lane_distance = abs(yellow_pos - round_pos) / 2
                        error = target_pos - FRAME_CENTER
                        steering = np.clip(Kp * error, -MAX_STEERING, MAX_STEERING)
                        throttle = BASE_THROTTLE

        # Send control
        controlData[0] = throttle
        controlData[1] = steering
        rgbServer.send(controlData.data)

        # Display
        rgb_display = draw_rgb_display(rgb_frame, seg_mask, yellow_pos, round_pos, target_pos, mode, all_detections)
        cv2.imshow('CSI + STRICT FUSION (BOTH MUST AGREE)', rgb_display)
        
        if show_depth:
            depth_display = draw_depth_display(depth_frame, obstacle_roi)
            cv2.imshow('RealSense Depth', depth_display)
        
        if show_lidar and lidar_frame_count > 0:
            cv2.imshow('LIDAR View', draw_lidar_cv())

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            emergency_stop = not emergency_stop
            print('\n🛑 EMERGENCY STOP!' if emergency_stop else '\n✅ Resumed')
        elif key == ord('v'):
            show_overlay = not show_overlay
        elif key == ord('d'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow('RealSense Depth')
        elif key == ord('l'):
            show_lidar = not show_lidar
            if not show_lidar:
                cv2.destroyWindow('LIDAR View')

        # Status printing
        if frameCount % 60 == 0 or stopped_for_obstacle:
            fusion_status = "BOTH!" if fusion_obstacle_detected else "CLEAR"
            single_sensor = ""
            if depth_obstacle_detected and not lidar_obstacle_detected:
                single_sensor = " (Depth only)"
            elif lidar_obstacle_detected and not depth_obstacle_detected:
                single_sensor = " (LIDAR only)"
            
            print(f'F:{frameCount} D:{depthFrameCount} L:{lidar_frame_count} | {mode} | Fusion:{fusion_status}{single_sensor} | T:{throttle:.3f} S:{steering:+.3f}')

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
    for _ in range(3):
        rgbServer.send(controlData.data)
        time.sleep(0.1)
    rgbServer.terminate()
    depthServer.terminate()
    lidarServer.terminate()
    cv2.destroyAllWindows()
    print('✅ Done!')