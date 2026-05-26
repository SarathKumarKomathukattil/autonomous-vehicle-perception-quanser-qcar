import cv2
import numpy as np
import time
import sys
from io import StringIO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

"""
HYBRID LANE FOLLOWER - ENHANCED KALMAN TRUST
=============================================
RULE 1: Cyan present → MEDIAN-BASED (No Kalman)
RULE 2: Cyan absent  → POLYNOMIAL-BASED (HIGH Kalman Trust)

ENHANCEMENT: Kalman filter has HIGHER PRIORITY over raw measurements
- Lower process noise = Trust Kalman model predictions more
- Higher measurement noise = Filter raw measurements more aggressively
- More weight given to Kalman-filtered trajectory
"""

class KalmanFilter:
    """
    Enhanced Kalman Filter with HIGHER TRUST for smooth tracking
    State: [x, y, vx, vy] - position and velocity
    """
    def __init__(self, process_noise=0.3, measurement_noise=8.0):
        """
        ENHANCED PARAMETERS:
        - Lower process_noise (0.3 instead of 0.5) = Trust model MORE
        - Higher measurement_noise (8.0 instead of 5.0) = Filter raw data MORE
        """
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.initialized = False
        
        # State transition matrix (constant velocity model)
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance (LOWER = trust model more)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.Q[2:, 2:] *= 1.5  # Slightly less noise for velocity
        
        # Measurement noise covariance (HIGHER = filter more)
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        
        # Innovation tracking for adaptive tuning
        self.innovation_history = []
        self.max_history = 10
        
        # Confidence tracking
        self.confidence = 0.0
        
    def initialize(self, measurement):
        """Initialize with first measurement"""
        self.state[0, 0] = measurement[0]
        self.state[1, 0] = measurement[1]
        self.state[2, 0] = 0.0
        self.state[3, 0] = 0.0
        self.initialized = True
        self.confidence = 0.5
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Increase confidence in prediction slightly
        self.confidence = min(0.95, self.confidence + 0.02)
        
        return self.state[:2].flatten()
    
    def update(self, measurement):
        """Update state with measurement"""
        if not self.initialized:
            self.initialize(measurement)
            return self.state[:2].flatten()
        
        # Measurement vector
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        # Innovation
        y = z - (self.H @ self.state)
        
        # Track innovation for adaptive noise
        innovation_mag = np.linalg.norm(y)
        self.innovation_history.append(innovation_mag)
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)
        
        # Adaptive measurement noise (MORE aggressive filtering)
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 1.0
        adaptive_factor = 1.0 + avg_innovation / 40.0  # More aggressive (was 50.0)
        adaptive_R = self.R * adaptive_factor
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + adaptive_R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        # Update confidence (higher with good measurements)
        if innovation_mag < 20:  # Low innovation = good prediction
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.confidence = max(0.3, self.confidence - 0.02)
        
        return self.state[:2].flatten()
    
    def get_confidence(self):
        """Get current confidence level"""
        return self.confidence
    
    def get_state(self):
        """Get current state"""
        if not self.initialized:
            return None
        return self.state[:2].flatten()
    
    def reset(self):
        """Reset filter"""
        self.state = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        self.initialized = False
        self.innovation_history = []
        self.confidence = 0.0


class HybridLaneFollower:
    def __init__(self):
        print("\n" + "="*70)
        print("HYBRID LANE FOLLOWER - ENHANCED KALMAN TRUST")
        print("="*70)
        
        print("\n🔌 Connecting to QLabs...")
        self.qlabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")
        print("✅ Connected!")
        
        self.qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        
        print("\n🚗 Spawning QCar...")
        self.qcar = QLabsQCar(self.qlabs)
        self.qcar.spawn_id(
            actorNumber=0,
            location=[-0.15, 3, 0.01],
            rotation=[0, 0, 300],
            waitForConfirmation=True
        )
        print("✅ QCar spawned!")
        
        print("\n▶️  Starting simulation...")
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        print("✅ Running!")
        
        # =================================================================
        # SHARED PARAMETERS
        # =================================================================
        self.speed_median = 0.036
        self.speed_polynomial = 0.031
        
        self.offset_from_white = 250
        self.offset_from_cyan = 250
        
        # =================================================================
        # MEDIAN-BASED PARAMETERS
        # =================================================================
        self.lower_cyan = np.array([80, 0, 150])
        self.upper_cyan = np.array([170, 255, 255])
        self.lower_white = np.array([0, 0, 240])
        self.upper_white = np.array([180, 25, 255])
        
        self.median_roi_start = 0.55
        self.median_roi_height = 0.45
        
        self.kp = 0.8
        self.kd = 0.2
        self.previous_error = 0
        
        # =================================================================
        # POLYNOMIAL-BASED PARAMETERS - ENHANCED KALMAN TRUST
        # =================================================================
        self.lower_road = np.array([0, 0, 50])
        self.upper_road = np.array([180, 80, 240])
        
        self.poly_roi_start = 0.60
        self.poly_roi_end = 0.95
        
        self.num_points = 10
        
        self.poly_degree = 2
        self.lookahead_y = 100
        
        self.steering_gain = 0.3
        self.max_steering = 0.5
        self.last_steering = 0.0
        
        # 🎯 ENHANCED KALMAN FILTER - HIGHER TRUST
        self.kalman = KalmanFilter(
            process_noise=0.3,      # LOWER = trust model predictions MORE
            measurement_noise=8.0   # HIGHER = filter raw measurements MORE
        )
        self.use_kalman = True
        self.frames_without_detection = 0
        self.max_prediction_frames = 8  # Allow more predictions (was 5)
        
        # 🎯 KALMAN PRIORITY WEIGHT
        # When blending Kalman with raw points, give MORE weight to Kalman
        self.kalman_weight = 3.0  # Kalman point counts 3x more than raw points
        
        # =================================================================
        # MODE TRACKING
        # =================================================================
        self.current_mode = "INITIALIZING"
        self.mode_switches = 0
        self.median_frames = 0
        self.poly_frames = 0
        self.kalman_updates = 0
        self.kalman_predictions = 0
        
        print(f"\n✅ Enhanced Kalman system initialized!")
        print(f"   Speed (MEDIAN): {self.speed_median}")
        print(f"   Speed (POLYNOMIAL): {self.speed_polynomial}")
        print(f"   🎯 ENHANCED Kalman Filter:")
        print(f"      • Process noise: 0.3 (LOWER = trust model MORE)")
        print(f"      • Measurement noise: 8.0 (HIGHER = filter MORE)")
        print(f"      • Kalman weight: {self.kalman_weight}x (prioritize filtered data)")
        print(f"      • Max predictions: {self.max_prediction_frames} frames")
        print(f"   RULE:")
        print(f"     🔵 Cyan present  → MEDIAN (No Kalman)")
        print(f"     🟣 Cyan absent   → POLYNOMIAL (HIGH Kalman Trust)")
        print("\n💡 Press 'q' to stop, 'k' to toggle Kalman\n")
    
    # =====================================================================
    # MEDIAN-BASED METHODS - NO KALMAN
    # =====================================================================
    
    def detect_lane_boundaries(self, frame):
        """Detect BOTH cyan (left) and white (right) boundaries"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        h, w = frame.shape[:2]
        roi_start_px = int(h * self.median_roi_start)
        roi_end_px = int(h * min(self.median_roi_start + self.median_roi_height, 1.0))
        
        cyan_roi = cyan_mask[roi_start_px:roi_end_px, :]
        white_roi = white_mask[roi_start_px:roi_end_px, :]
        
        cyan_center = None
        M_cyan = cv2.moments(cyan_roi)
        if M_cyan['m00'] > 100:
            cx = int(M_cyan['m10'] / M_cyan['m00'])
            cy = int(M_cyan['m01'] / M_cyan['m00']) + roi_start_px
            cyan_center = (cx, cy)
        
        white_center = None
        M_white = cv2.moments(white_roi)
        if M_white['m00'] > 100:
            cx = int(M_white['m10'] / M_white['m00'])
            cy = int(M_white['m01'] / M_white['m00']) + roi_start_px
            white_center = (cx, cy)
        
        lane_center = None
        strategy = "NONE"
        
        if cyan_center and white_center:
            center_x = (cyan_center[0] + white_center[0]) // 2
            center_y = (cyan_center[1] + white_center[1]) // 2
            lane_center = (center_x, center_y)
            strategy = "BOTH"
        elif white_center:
            target_x = white_center[0] - self.offset_from_white
            target_y = white_center[1]
            lane_center = (target_x, target_y)
            strategy = "WHITE_ONLY"
        elif cyan_center:
            target_x = cyan_center[0] + self.offset_from_cyan
            target_y = cyan_center[1]
            lane_center = (target_x, target_y)
            strategy = "CYAN_ONLY"
        
        cyan_pixels = cv2.countNonZero(cyan_roi)
        white_pixels = cv2.countNonZero(white_roi)
        
        return lane_center, cyan_center, white_center, cyan_pixels, white_pixels, strategy
    
    def calculate_median_steering(self, lane_center, width):
        """Calculate steering using PD controller"""
        if lane_center is None:
            self.previous_error = 0
            return 0.0
        
        center_x = width // 2
        error = (lane_center[0] - center_x) / center_x
        
        derivative = error - self.previous_error
        steering = -self.kp * error - self.kd * derivative
        self.previous_error = error
        
        return np.clip(steering, -0.3, 0.3)
    
    # =====================================================================
    # POLYNOMIAL-BASED METHODS - ENHANCED KALMAN
    # =====================================================================
    
    def create_road_mask(self, frame):
        """Create road mask"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.lower_road, self.upper_road)
        
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def extract_center_points(self, road_mask):
        """Extract center points with intelligent strategy"""
        h, w = road_mask.shape
        
        roi_start_px = int(h * self.poly_roi_start)
        roi_end_px = int(h * self.poly_roi_end)
        
        road_roi = road_mask[roi_start_px:roi_end_px, :]
        
        roi_height = roi_end_px - roi_start_px
        sample_rows = np.linspace(0, roi_height - 1, self.num_points, dtype=int)
        
        center_points = []
        strategy_used = "UNKNOWN"
        
        for row_idx in sample_rows:
            road_row = road_roi[row_idx, :]
            road_row_filtered = road_row.copy()
            
            edge_margin = 40
            road_row_filtered[:edge_margin] = 0
            road_row_filtered[-edge_margin:] = 0
            
            road_pixels = np.where(road_row_filtered > 0)[0]
            
            if len(road_pixels) > 5:
                left_edge = np.min(road_pixels)
                right_edge = np.max(road_pixels)
                
                left_gap = left_edge - edge_margin
                right_gap = (w - edge_margin) - right_edge
                
                MIN_GAP_THRESHOLD = 80
                
                if left_gap > MIN_GAP_THRESHOLD and right_gap > MIN_GAP_THRESHOLD:
                    center_x = (left_edge + right_edge) // 2
                    strategy_used = "BOTH_EDGES"
                else:
                    center_x = right_edge - self.offset_from_white
                    
                    if center_x < left_edge:
                        center_x = (left_edge + right_edge) // 2
                    
                    strategy_used = "RIGHT_EDGE_ONLY"
                
                center_y = row_idx + roi_start_px
                center_points.append((center_x, center_y))
        
        return center_points, strategy_used
    
    def apply_kalman_filter(self, center_points):
        """
        🎯 Apply ENHANCED Kalman filter - PRIORITIZE filtered output
        """
        if not self.use_kalman:
            if len(center_points) > 0:
                avg_x = int(np.mean([p[0] for p in center_points]))
                avg_y = int(np.mean([p[1] for p in center_points]))
                return (avg_x, avg_y), "RAW", 0.0
            return None, "NO_DETECTION", 0.0
        
        # Kalman enabled
        if len(center_points) > 0:
            # Calculate weighted average (emphasize closer points)
            weights = np.linspace(0.5, 1.5, len(center_points))
            weighted_x = np.average([p[0] for p in center_points], weights=weights)
            weighted_y = np.average([p[1] for p in center_points], weights=weights)
            raw_measurement = (weighted_x, weighted_y)
            
            # Update Kalman with measurement
            filtered_pos = self.kalman.update(raw_measurement)
            filtered_point = (int(filtered_pos[0]), int(filtered_pos[1]))
            
            self.frames_without_detection = 0
            self.kalman_updates += 1
            
            confidence = self.kalman.get_confidence()
            return filtered_point, "UPDATED", confidence
        
        else:
            # No measurement - use prediction (now allows MORE frames)
            self.frames_without_detection += 1
            
            if self.frames_without_detection <= self.max_prediction_frames:
                # Use Kalman prediction
                predicted_pos = self.kalman.predict()
                filtered_point = (int(predicted_pos[0]), int(predicted_pos[1]))
                
                self.kalman_predictions += 1
                confidence = self.kalman.get_confidence()
                return filtered_point, f"PREDICTED-{self.frames_without_detection}", confidence
            else:
                # Too many frames without detection
                self.kalman.reset()
                return None, "RESET", 0.0
    
    def fit_polynomial_with_enhanced_kalman(self, filtered_point, raw_points, confidence):
        """
        🎯 ENHANCED: Fit polynomial with HIGHER WEIGHT on Kalman-filtered point
        The filtered point is given MORE importance than raw points
        """
        if filtered_point is None:
            return None
        
        if len(raw_points) >= 2:
            # Blend with HEAVY emphasis on Kalman-filtered point
            x_coords = [p[0] for p in raw_points]
            y_coords = [p[1] for p in raw_points]
            
            # Add Kalman point MULTIPLE times (higher weight)
            # Weight increases with confidence
            num_kalman_copies = int(self.kalman_weight * (1 + confidence))  # 3-6 copies
            
            for _ in range(num_kalman_copies):
                x_coords.append(filtered_point[0])
                y_coords.append(filtered_point[1])
            
            x_coords = np.array(x_coords, dtype=np.float32)
            y_coords = np.array(y_coords, dtype=np.float32)
            
            # Fit polynomial with heavy Kalman influence
            coefficients = np.polyfit(y_coords, x_coords, self.poly_degree)
        else:
            # Not enough raw points - use Kalman-only polynomial
            coefficients = np.array([0, 0, filtered_point[0]], dtype=np.float32)
        
        return coefficients
    
    def calculate_steering_angle(self, coefficients, frame_height, frame_width):
        """Calculate steering angle from polynomial"""
        if coefficients is None:
            return 0.0, None
        
        target_y = frame_height - self.lookahead_y
        target_y = max(0, min(frame_height - 1, target_y))
        
        target_x = np.polyval(coefficients, np.array([target_y]))[0]
        
        ref_x = frame_width // 2
        ref_y = frame_height - 1
        
        dx = target_x - ref_x
        dy = ref_y - target_y
        
        steering_angle = np.arctan2(dx, dy) * 180 / np.pi
        
        return steering_angle, (int(target_x), int(target_y))
    
    def angle_to_steering_command(self, steering_angle):
        """Convert steering angle to steering command"""
        max_angle = 45.0
        steering_cmd = np.clip(steering_angle / max_angle, -1.0, 1.0)
        steering_cmd *= self.steering_gain
        steering_cmd = np.clip(steering_cmd, -self.max_steering, self.max_steering)
        
        steering_cmd = -steering_cmd
        
        # Slightly more aggressive smoothing with high-confidence Kalman
        alpha = 0.6  # More responsive (was 0.5)
        steering_cmd = alpha * steering_cmd + (1 - alpha) * self.last_steering
        self.last_steering = steering_cmd
        
        return steering_cmd
    
    # =====================================================================
    # HYBRID PROCESSING
    # =====================================================================
    
    def process_frame_hybrid(self, frame):
        """Process frame with enhanced Kalman priority"""
        h, w = frame.shape[:2]
        
        # Check for cyan presence
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        cyan_pixels_total = cv2.countNonZero(cyan_mask)
        
        CYAN_THRESHOLD = 100
        cyan_present = cyan_pixels_total > CYAN_THRESHOLD
        
        steering = 0.0
        throttle = 0.0
        mode = "UNKNOWN"
        vis_data = {}
        
        # MODE 1: MEDIAN-BASED (No Kalman)
        if cyan_present:
            mode = "MEDIAN"
            
            if self.current_mode == "POLYNOMIAL":
                self.kalman.reset()
            
            result = self.detect_lane_boundaries(frame)
            lane_center, cyan_center, white_center, cyan_px, white_px, strategy = result
            
            steering = self.calculate_median_steering(lane_center, w)
            throttle = self.speed_median
            
            vis_data = {
                'mode': 'MEDIAN',
                'lane_center': lane_center,
                'cyan_center': cyan_center,
                'white_center': white_center,
                'cyan_pixels': cyan_px,
                'white_pixels': white_px,
                'strategy': strategy,
                'raw_points': None,
                'filtered_point': None,
                'center_points': None,
                'coefficients': None,
                'steering_angle': None,
                'target_point': None,
                'kalman_status': 'N/A',
                'kalman_confidence': 0.0
            }
            
            self.median_frames += 1
        
        # MODE 2: POLYNOMIAL-BASED (Enhanced Kalman)
        else:
            mode = "POLYNOMIAL"
            
            road_mask = self.create_road_mask(frame)
            raw_points, poly_strategy = self.extract_center_points(road_mask)
            
            # Apply ENHANCED Kalman filter
            filtered_point, kalman_status, confidence = self.apply_kalman_filter(raw_points)
            
            # Fit polynomial with HIGH Kalman weight
            if self.use_kalman and filtered_point is not None:
                coefficients = self.fit_polynomial_with_enhanced_kalman(
                    filtered_point, raw_points, confidence
                )
            else:
                # Fallback without Kalman
                if len(raw_points) >= 3:
                    x_coords = np.array([p[0] for p in raw_points], dtype=np.float32)
                    y_coords = np.array([p[1] for p in raw_points], dtype=np.float32)
                    coefficients = np.polyfit(y_coords, x_coords, self.poly_degree)
                else:
                    coefficients = None
            
            # Calculate steering
            steering_angle, target_point = self.calculate_steering_angle(coefficients, h, w)
            
            if coefficients is not None:
                steering = self.angle_to_steering_command(steering_angle)
                throttle = self.speed_polynomial
            else:
                steering = 0.0
                throttle = 0.0
                steering_angle = 0.0
            
            vis_data = {
                'mode': 'POLYNOMIAL',
                'lane_center': None,
                'cyan_center': None,
                'white_center': None,
                'cyan_pixels': cyan_pixels_total,
                'white_pixels': 0,
                'strategy': f"{len(raw_points)}pts-{poly_strategy}",
                'raw_points': raw_points,
                'filtered_point': filtered_point,
                'center_points': raw_points,
                'coefficients': coefficients,
                'steering_angle': steering_angle,
                'target_point': target_point,
                'kalman_status': kalman_status,
                'kalman_confidence': confidence
            }
            
            self.poly_frames += 1
        
        # Track mode switches
        if mode != self.current_mode and self.current_mode != "INITIALIZING":
            self.mode_switches += 1
            print(f"\n🔄 MODE SWITCH: {self.current_mode} → {mode}")
        
        self.current_mode = mode
        
        return steering, throttle, vis_data
    
    def visualize_hybrid(self, frame, steering, throttle, vis_data):
        """Visualize with enhanced Kalman indicators"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        mode = vis_data['mode']
        
        # MEDIAN MODE VISUALIZATION
        if mode == 'MEDIAN':
            if vis_data['cyan_center']:
                cv2.circle(vis, vis_data['cyan_center'], 10, (255, 255, 0), -1)
                cv2.circle(vis, vis_data['cyan_center'], 13, (0, 0, 0), 2)
                cv2.putText(vis, "CYAN", (vis_data['cyan_center'][0]-20, vis_data['cyan_center'][1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            if vis_data['white_center']:
                cv2.circle(vis, vis_data['white_center'], 10, (255, 255, 255), -1)
                cv2.circle(vis, vis_data['white_center'], 13, (0, 0, 0), 2)
                cv2.putText(vis, "WHITE", (vis_data['white_center'][0]-25, vis_data['white_center'][1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if vis_data['lane_center']:
                cv2.circle(vis, vis_data['lane_center'], 15, (0, 255, 0), -1)
                cv2.circle(vis, vis_data['lane_center'], 18, (0, 0, 0), 2)
                cv2.line(vis, (w//2, vis_data['lane_center'][1]), vis_data['lane_center'], (0, 255, 0), 3)
            
            cv2.line(vis, (w//2, 0), (w//2, h), (255, 0, 0), 2)
            
            mode_color = (0, 255, 255)
            mode_text = "MODE: MEDIAN (No Kalman)"
        
        # POLYNOMIAL MODE WITH ENHANCED KALMAN
        else:
            # Draw raw points (smaller, dimmer)
            if vis_data['raw_points']:
                for pt in vis_data['raw_points']:
                    cv2.circle(vis, pt, 3, (0, 0, 180), -1)  # Dimmer red
            
            # Draw KALMAN point (LARGER, BRIGHTER - more emphasis)
            if vis_data['filtered_point']:
                # Confidence-based visualization
                confidence = vis_data['kalman_confidence']
                brightness = int(155 + 100 * confidence)  # 155-255
                
                cv2.circle(vis, vis_data['filtered_point'], 15, (0, brightness, 0), -1)  # Larger
                cv2.circle(vis, vis_data['filtered_point'], 18, (255, 255, 255), 3)  # Thicker outline
                
                # Show confidence
                conf_text = f"KALMAN {confidence:.0%}"
                cv2.putText(vis, conf_text, 
                           (vis_data['filtered_point'][0]-40, vis_data['filtered_point'][1]-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw polynomial curve (thicker with Kalman)
            if vis_data['coefficients'] is not None and vis_data['center_points']:
                y_min = min(pt[1] for pt in vis_data['center_points'])
                y_max = max(pt[1] for pt in vis_data['center_points'])
                
                y_curve = np.linspace(y_min, y_max, 100)
                x_curve = np.polyval(vis_data['coefficients'], y_curve)
                
                curve_points = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
                cv2.polylines(vis, [curve_points], False, (0, 255, 0), 5)  # Thicker
                
                if vis_data['target_point']:
                    cv2.circle(vis, vis_data['target_point'], 12, (255, 0, 255), -1)
                    ref_point = (w // 2, h - 1)
                    cv2.line(vis, ref_point, vis_data['target_point'], (255, 255, 0), 3)
                    cv2.circle(vis, ref_point, 8, (0, 255, 255), -1)
            
            mode_color = (255, 0, 255)
            kalman_text = f" [Kalman: {vis_data['kalman_status']}]" if self.use_kalman else " [Kalman: OFF]"
            mode_text = f"MODE: POLYNOMIAL (ENHANCED){kalman_text}"
        
        # STATUS OVERLAY
        overlay_height = 230 if mode == 'POLYNOMIAL' else 180
        cv2.rectangle(vis, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(vis, (0, 0), (w, overlay_height), mode_color, 3)
        
        cv2.putText(vis, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        cv2.putText(vis, f"Steering: {steering:.3f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Throttle: {throttle:.3f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Strategy: {vis_data['strategy']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Cyan: {vis_data['cyan_pixels']}px", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Enhanced Kalman statistics
        if mode == 'POLYNOMIAL' and self.use_kalman:
            confidence = vis_data['kalman_confidence']
            conf_color = (0, int(155 + 100*confidence), 0)
            
            cv2.putText(vis, f"K-Confidence: {confidence:.0%}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
            cv2.putText(vis, f"K-Updates: {self.kalman_updates}", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis, f"K-Predictions: {self.kalman_predictions}", (10, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        cv2.putText(vis, f"Switches: {self.mode_switches}", (w-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Steering arrow
        steer_x = int(w//2 + (steering * w//4))
        cv2.arrowedLine(vis, (w//2, h-50), (steer_x, h-50), (0, 255, 255), 4)
        
        return vis
    
    def run(self):
        """Main loop with enhanced Kalman"""
        
        print("🔧 Initializing QCar hardware...")
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            use_hardware = True
            print("✅ Hardware enabled")
        except Exception as e:
            sys.stdin = old_stdin
            use_hardware = False
            print(f"⚠️  Visualization only: {e}")
        
        print("\n🚗 ENHANCED KALMAN SYSTEM ACTIVE!\n")
        
        frame_count = 0
        
        try:
            while True:
                frame_count += 1
                
                success, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success and image_data is not None:
                    try:
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        img = img.reshape((410, 820, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        steering, throttle, vis_data = self.process_frame_hybrid(img)
                        
                        if use_hardware:
                            qcar_hw.write(throttle, steering)
                        
                        vis = self.visualize_hybrid(img, steering, throttle, vis_data)
                        
                        cv2.imshow('HYBRID - Enhanced Kalman Trust', vis)
                        
                        if frame_count % 30 == 0:
                            mode_emoji = "🔵" if vis_data['mode'] == 'MEDIAN' else "🟣"
                            hw = "🚗" if use_hardware else "👁️"
                            
                            if vis_data['mode'] == 'POLYNOMIAL':
                                conf = vis_data['kalman_confidence']
                                k_info = f"K:{vis_data['kalman_status']} C:{conf:.0%}"
                            else:
                                k_info = ""
                            
                            print(f"Frame {frame_count}: {mode_emoji} {vis_data['mode']} {k_info} | "
                                  f"{hw} | Steer: {steering:.3f}")
                        
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"⚠️  Error: {e}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('k'):
                    self.use_kalman = not self.use_kalman
                    status = "ENABLED" if self.use_kalman else "DISABLED"
                    print(f"\n🔄 Enhanced Kalman {status}")
                    if not self.use_kalman:
                        self.kalman.reset()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            print("\n🛑 Stopping...")
            if use_hardware:
                qcar_hw.write(0.0, 0.0)
                time.sleep(0.5)
            
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            print(f"\n📊 Statistics:")
            print(f"   Frames: {frame_count}")
            print(f"   Median: {self.median_frames} ({self.median_frames/max(frame_count,1)*100:.1f}%)")
            print(f"   Polynomial: {self.poly_frames} ({self.poly_frames/max(frame_count,1)*100:.1f}%)")
            print(f"   Switches: {self.mode_switches}")
            if self.kalman_updates > 0 or self.kalman_predictions > 0:
                total = self.kalman_updates + self.kalman_predictions
                print(f"\n🎯 Enhanced Kalman Stats:")
                print(f"   Updates: {self.kalman_updates}")
                print(f"   Predictions: {self.kalman_predictions}")
                print(f"   Prediction Rate: {self.kalman_predictions/total*100:.1f}%")
            print("\n✅ Done!\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HYBRID LANE FOLLOWER - ENHANCED KALMAN TRUST")
    print("="*70)
    print("\n🎯 KEY ENHANCEMENTS:")
    print("   ✅ LOWER process noise (0.3) = Trust Kalman model MORE")
    print("   ✅ HIGHER measurement noise (8.0) = Filter raw data MORE")
    print("   ✅ Kalman point weighted 3x-6x over raw points")
    print("   ✅ Longer predictions (up to 8 frames)")
    print("   ✅ Confidence tracking for adaptive behavior")
    print("\n📋 OPERATION:")
    print("   🔵 Cyan present  → MEDIAN (No Kalman)")
    print("   🟣 Cyan absent   → POLYNOMIAL (HIGH Kalman Trust)")
    print("\n💡 CONTROLS:")
    print("   • 'q' - Quit")
    print("   • 'k' - Toggle Kalman ON/OFF")
    print("\n🔧 FURTHER TUNING (if needed):")
    print("   • Increase kalman_weight (line 118): 3.0 → 5.0")
    print("   • Decrease process_noise (line 113): 0.3 → 0.2")
    print("   • Increase measurement_noise (line 114): 8.0 → 10.0")
    print("="*70 + "\n")
    
    input("Press ENTER to start enhanced Kalman system... ")
    
    follower = HybridLaneFollower()
    follower.run()