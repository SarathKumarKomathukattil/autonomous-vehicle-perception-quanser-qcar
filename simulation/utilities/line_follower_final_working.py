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
HYBRID LANE FOLLOWER - KALMAN FILTER IN POLYNOMIAL MODE ONLY
=============================================================
RULE 1: Cyan present → MEDIAN-BASED (Code 1) - NO KALMAN
RULE 2: Cyan absent  → POLYNOMIAL-BASED (Code 2) - WITH KALMAN FILTER

Kalman filter ONLY active in polynomial mode to smooth roundabout performance
"""

class KalmanFilter:
    """
    Kalman Filter for smooth lane center tracking in polynomial mode
    State: [x, y, vx, vy] - position and velocity
    """
    def __init__(self, process_noise=1.0, measurement_noise=10.0):
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
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.Q[2:, 2:] *= 2.0  # Higher noise for velocity
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.P = np.eye(4, dtype=np.float32) * 1000.0
        
        # Innovation tracking for adaptive tuning
        self.innovation_history = []
        self.max_history = 10
        
    def initialize(self, measurement):
        """Initialize with first measurement"""
        self.state[0, 0] = measurement[0]
        self.state[1, 0] = measurement[1]
        self.state[2, 0] = 0.0
        self.state[3, 0] = 0.0
        self.initialized = True
        
    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
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
        
        # Adaptive measurement noise
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 1.0
        adaptive_R = self.R * (1.0 + avg_innovation / 50.0)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + adaptive_R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:2].flatten()
    
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


class HybridLaneFollower:
    def __init__(self):
        print("\n" + "="*70)
        print("HYBRID LANE FOLLOWER - KALMAN IN POLYNOMIAL MODE ONLY")
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
        self.speed_median = 0.036      # Normal speed when cyan visible
        self.speed_polynomial = 0.031  # Slower in polynomial mode
        
        self.offset_from_white = 250
        self.offset_from_cyan = 250
        
        # =================================================================
        # MEDIAN-BASED (CODE 1) PARAMETERS - NO KALMAN
        # =================================================================
        self.lower_cyan = np.array([85, 80, 150])    
        self.upper_cyan = np.array([95, 255, 255])   
        self.lower_white = np.array([0, 0, 250])     
        self.upper_white = np.array([180, 20, 255])  
        self.median_roi_start = 0.70  
        self.median_roi_height = 0.30
        
        self.kp = 0.8
        self.kd = 0.2
        self.previous_error = 0
        
        # =================================================================
        # POLYNOMIAL-BASED (CODE 2) PARAMETERS - WITH KALMAN
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
        
        # 🎯 KALMAN FILTER - ONLY FOR POLYNOMIAL MODE
        self.kalman = KalmanFilter(process_noise=0.5, measurement_noise=5.0)
        self.use_kalman = True  # Toggle with 'k' key
        self.frames_without_detection = 0
        self.max_prediction_frames = 5
        
        # =================================================================
        # MODE TRACKING
        # =================================================================
        self.current_mode = "INITIALIZING"
        self.mode_switches = 0
        self.median_frames = 0
        self.poly_frames = 0
        self.kalman_updates = 0
        self.kalman_predictions = 0
        
        print(f"\n✅ Hybrid system initialized!")
        print(f"   Speed (MEDIAN mode): {self.speed_median}")
        print(f"   Speed (POLYNOMIAL mode): {self.speed_polynomial}")
        print(f"   Offset from right edge: {self.offset_from_white}px")
        print(f"   🎯 Kalman Filter: ENABLED (Polynomial mode ONLY)")
        print(f"   STRICT RULE:")
        print(f"     🔵 Cyan detected → MEDIAN-BASED (No Kalman)")
        print(f"     🟣 Cyan absent   → POLYNOMIAL-BASED (With Kalman)")
        print("\n💡 Press 'q' to stop, 'k' to toggle Kalman\n")
    
    # =====================================================================
    # MEDIAN-BASED METHODS (CODE 1) - NO KALMAN
    # =====================================================================
    
    def detect_lane_boundaries(self, frame):
        """Detect BOTH cyan (left) and white (right) boundaries - MEDIAN METHOD"""
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
        """Calculate steering using PD controller - MEDIAN METHOD"""
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
    # POLYNOMIAL-BASED METHODS (CODE 2) - WITH KALMAN
    # =====================================================================
    
    def create_road_mask(self, frame):
        """Create road mask - POLYNOMIAL METHOD"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.lower_road, self.upper_road)
        
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def extract_center_points(self, road_mask):
        """
        Extract center points - INTELLIGENT STRATEGY
        
        Strategy:
        1. If BOTH left AND right edges detected → Use TRUE CENTER (best for roundabouts!)
        2. If ONLY right edge detected → Use OFFSET from right edge (median strategy)
        """
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
                    # ✅ BOTH EDGES VISIBLE - Use TRUE CENTER
                    center_x = (left_edge + right_edge) // 2
                    strategy_used = "BOTH_EDGES"
                else:
                    # ⚠️ ONLY RIGHT EDGE VISIBLE - Use offset strategy
                    center_x = right_edge - self.offset_from_white
                    
                    if center_x < left_edge:
                        center_x = (left_edge + right_edge) // 2
                    
                    strategy_used = "RIGHT_EDGE_ONLY"
                
                center_y = row_idx + roi_start_px
                center_points.append((center_x, center_y))
        
        return center_points, strategy_used
    
    def apply_kalman_filter(self, center_points):
        """
        🎯 Apply Kalman filter to center points - POLYNOMIAL MODE ONLY
        Returns: filtered_point, kalman_status
        """
        if not self.use_kalman:
            # Kalman disabled - use raw average
            if len(center_points) > 0:
                avg_x = int(np.mean([p[0] for p in center_points]))
                avg_y = int(np.mean([p[1] for p in center_points]))
                return (avg_x, avg_y), "RAW"
            return None, "NO_DETECTION"
        
        # Kalman enabled
        if len(center_points) > 0:
            # Calculate weighted average (more weight to closer points)
            weights = np.linspace(0.5, 1.5, len(center_points))
            weighted_x = np.average([p[0] for p in center_points], weights=weights)
            weighted_y = np.average([p[1] for p in center_points], weights=weights)
            raw_measurement = (weighted_x, weighted_y)
            
            # Update Kalman with measurement
            filtered_pos = self.kalman.update(raw_measurement)
            filtered_point = (int(filtered_pos[0]), int(filtered_pos[1]))
            
            self.frames_without_detection = 0
            self.kalman_updates += 1
            return filtered_point, "UPDATED"
        
        else:
            # No measurement - use prediction
            self.frames_without_detection += 1
            
            if self.frames_without_detection <= self.max_prediction_frames:
                # Use Kalman prediction
                predicted_pos = self.kalman.predict()
                filtered_point = (int(predicted_pos[0]), int(predicted_pos[1]))
                
                self.kalman_predictions += 1
                return filtered_point, f"PREDICTED-{self.frames_without_detection}"
            else:
                # Too many frames without detection
                self.kalman.reset()
                return None, "RESET"
    
    def fit_polynomial(self, center_points):
        """Fit 2nd degree polynomial to center points - WITHOUT KALMAN"""
        if len(center_points) < 3:
            return None
        
        x_coords = np.array([pt[0] for pt in center_points], dtype=np.float32)
        y_coords = np.array([pt[1] for pt in center_points], dtype=np.float32)
        
        coefficients = np.polyfit(y_coords, x_coords, self.poly_degree)
        return coefficients
    
    def fit_polynomial_with_kalman(self, filtered_point, raw_points):
        """Fit polynomial using Kalman-filtered point"""
        if filtered_point is None:
            return None
        
        # Use filtered point as primary anchor
        if len(raw_points) >= 2:
            # Blend raw points with filtered point
            x_coords = [p[0] for p in raw_points]
            y_coords = [p[1] for p in raw_points]
            
            # Add filtered point with emphasis
            x_coords.append(filtered_point[0])
            y_coords.append(filtered_point[1])
            
            x_coords = np.array(x_coords, dtype=np.float32)
            y_coords = np.array(y_coords, dtype=np.float32)
            
            coefficients = np.polyfit(y_coords, x_coords, self.poly_degree)
        else:
            # Create synthetic polynomial from filtered point
            coefficients = np.array([0, 0, filtered_point[0]], dtype=np.float32)
        
        return coefficients
    
    def calculate_steering_angle(self, coefficients, frame_height, frame_width):
        """Calculate steering angle from polynomial - POLYNOMIAL METHOD"""
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
        """Convert steering angle to steering command - POLYNOMIAL METHOD"""
        max_angle = 45.0
        steering_cmd = np.clip(steering_angle / max_angle, -1.0, 1.0)
        steering_cmd *= self.steering_gain
        steering_cmd = np.clip(steering_cmd, -self.max_steering, self.max_steering)
        
        # INVERT steering
        steering_cmd = -steering_cmd
        
        alpha = 0.5
        steering_cmd = alpha * steering_cmd + (1 - alpha) * self.last_steering
        self.last_steering = steering_cmd
        
        return steering_cmd
    
    # =====================================================================
    # HYBRID DECISION & VISUALIZATION
    # =====================================================================
    
    def process_frame_hybrid(self, frame):
        """
        STRICT MODE SELECTION:
        1. Check if cyan line is present
        2. If YES → Use MEDIAN-BASED (NO KALMAN)
        3. If NO  → Use POLYNOMIAL-BASED (WITH KALMAN)
        """
        h, w = frame.shape[:2]
        
        # Check for cyan line presence
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        cyan_pixels_total = cv2.countNonZero(cyan_mask)
        
        CYAN_THRESHOLD = 100
        cyan_present = cyan_pixels_total > CYAN_THRESHOLD
        
        steering = 0.0
        throttle = 0.0
        mode = "UNKNOWN"
        vis_data = {}
        
        # ============================================================
        # MODE 1: MEDIAN-BASED (When cyan is present) - NO KALMAN
        # ============================================================
        if cyan_present:
            mode = "MEDIAN"
            
            # Reset Kalman when switching to median mode
            if self.current_mode == "POLYNOMIAL":
                self.kalman.reset()
                print("   🔄 Kalman filter RESET (entering median mode)")
            
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
                'kalman_status': 'N/A'
            }
            
            self.median_frames += 1
        
        # ============================================================
        # MODE 2: POLYNOMIAL-BASED (When cyan is absent) - WITH KALMAN
        # ============================================================
        else:
            mode = "POLYNOMIAL"
            
            road_mask = self.create_road_mask(frame)
            raw_points, poly_strategy = self.extract_center_points(road_mask)
            
            # 🎯 Apply Kalman filter (ONLY in polynomial mode)
            filtered_point, kalman_status = self.apply_kalman_filter(raw_points)
            
            # Fit polynomial
            if self.use_kalman and filtered_point is not None:
                coefficients = self.fit_polynomial_with_kalman(filtered_point, raw_points)
            else:
                coefficients = self.fit_polynomial(raw_points)
            
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
                'kalman_status': kalman_status
            }
            
            self.poly_frames += 1
        
        # Track mode switches
        if mode != self.current_mode and self.current_mode != "INITIALIZING":
            self.mode_switches += 1
            print(f"\n🔄 MODE SWITCH: {self.current_mode} → {mode}")
        
        self.current_mode = mode
        
        return steering, throttle, vis_data
    
    def visualize_hybrid(self, frame, steering, throttle, vis_data):
        """Visualize with mode-specific overlays"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        mode = vis_data['mode']
        
        # ============================================================
        # MEDIAN-BASED VISUALIZATION (NO KALMAN)
        # ============================================================
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
            mode_text = "MODE: MEDIAN-BASED (Cyan Present) [No Kalman]"
        
        # ============================================================
        # POLYNOMIAL-BASED VISUALIZATION (WITH KALMAN)
        # ============================================================
        else:
            # Draw RAW center points (small red dots)
            if vis_data['raw_points']:
                for pt in vis_data['raw_points']:
                    cv2.circle(vis, pt, 4, (0, 0, 255), -1)
            
            # Draw KALMAN-FILTERED point (larger green dot)
            if vis_data['filtered_point']:
                cv2.circle(vis, vis_data['filtered_point'], 12, (0, 255, 0), -1)
                cv2.circle(vis, vis_data['filtered_point'], 15, (255, 255, 255), 2)
                cv2.putText(vis, "KALMAN", (vis_data['filtered_point'][0]-30, vis_data['filtered_point'][1]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw polynomial curve
            if vis_data['coefficients'] is not None and vis_data['center_points']:
                y_min = min(pt[1] for pt in vis_data['center_points'])
                y_max = max(pt[1] for pt in vis_data['center_points'])
                
                y_curve = np.linspace(y_min, y_max, 100)
                x_curve = np.polyval(vis_data['coefficients'], y_curve)
                
                curve_points = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)])
                cv2.polylines(vis, [curve_points], False, (0, 255, 0), 4)
                
                if vis_data['target_point']:
                    cv2.circle(vis, vis_data['target_point'], 12, (255, 0, 255), -1)
                    ref_point = (w // 2, h - 1)
                    cv2.line(vis, ref_point, vis_data['target_point'], (255, 255, 0), 2)
                    cv2.circle(vis, ref_point, 8, (0, 255, 255), -1)
            
            mode_color = (255, 0, 255)
            kalman_text = f" [Kalman: {vis_data['kalman_status']}]" if self.use_kalman else " [Kalman: OFF]"
            
            if "BOTH_EDGES" in vis_data['strategy']:
                mode_text = f"MODE: POLYNOMIAL (BOTH edges){kalman_text}"
            else:
                mode_text = f"MODE: POLYNOMIAL (Right edge){kalman_text}"
        
        # ============================================================
        # COMMON STATUS OVERLAY
        # ============================================================
        overlay_height = 210 if mode == 'POLYNOMIAL' else 180
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
        
        # Kalman statistics (only in polynomial mode)
        if mode == 'POLYNOMIAL' and self.use_kalman:
            cv2.putText(vis, f"K-Updates: {self.kalman_updates}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis, f"K-Predictions: {self.kalman_predictions}", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        cv2.putText(vis, f"Switches: {self.mode_switches}", (w-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Steering arrow
        steer_x = int(w//2 + (steering * w//4))
        cv2.arrowedLine(vis, (w//2, h-50), (steer_x, h-50), (0, 255, 255), 4)
        
        return vis
    
    def run(self):
        """Main hybrid loop"""
        
        print("🔧 Initializing QCar hardware interface...")
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            use_hardware = True
            print("✅ QCar hardware control enabled (QCar1)")
        except Exception as e:
            sys.stdin = old_stdin
            use_hardware = False
            print(f"⚠️  No hardware control: {e}")
        
        print("\n🚗 HYBRID LANE FOLLOWING ACTIVE!\n")
        print("   🔵 MEDIAN mode: No Kalman (cyan present)")
        print("   🟣 POLYNOMIAL mode: WITH Kalman (cyan absent)\n")
        
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
                        
                        # HYBRID PROCESSING
                        steering, throttle, vis_data = self.process_frame_hybrid(img)
                        
                        if use_hardware:
                            qcar_hw.write(throttle, steering)
                        
                        vis = self.visualize_hybrid(img, steering, throttle, vis_data)
                        
                        cv2.imshow('HYBRID Lane Follower - Kalman in Polynomial Only', vis)
                        
                        if frame_count % 30 == 0:
                            mode_emoji = "🔵" if vis_data['mode'] == 'MEDIAN' else "🟣"
                            hw_status = "🚗 DRIVING" if use_hardware else "👁️ VIZ"
                            k_status = f"K:{vis_data['kalman_status']}" if vis_data['mode'] == 'POLYNOMIAL' else ""
                            print(f"Frame {frame_count}: {mode_emoji} {vis_data['mode']} {k_status} | "
                                  f"{hw_status} | Steer: {steering:.3f} | "
                                  f"Cyan: {vis_data['cyan_pixels']}px")
                        
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"⚠️  Error: {e}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('k'):
                    self.use_kalman = not self.use_kalman
                    status = "ENABLED" if self.use_kalman else "DISABLED"
                    print(f"\n🔄 Kalman filter {status} (polynomial mode only)")
                    if not self.use_kalman:
                        self.kalman.reset()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            print("\n🛑 Stopping car...")
            if use_hardware:
                qcar_hw.write(0.0, 0.0)
                time.sleep(0.5)
            
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            print(f"\n📊 Session Statistics:")
            print(f"   Total Frames: {frame_count}")
            print(f"   Median Frames: {self.median_frames} ({self.median_frames/max(frame_count,1)*100:.1f}%)")
            print(f"   Polynomial Frames: {self.poly_frames} ({self.poly_frames/max(frame_count,1)*100:.1f}%)")
            print(f"   Mode Switches: {self.mode_switches}")
            if self.kalman_updates > 0 or self.kalman_predictions > 0:
                total = self.kalman_updates + self.kalman_predictions
                print(f"\n🎯 Kalman Filter Statistics (Polynomial mode only):")
                print(f"   Updates: {self.kalman_updates}")
                print(f"   Predictions: {self.kalman_predictions}")
                print(f"   Prediction Rate: {self.kalman_predictions/total*100:.1f}%")
            print("\n✅ Done!\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HYBRID LANE FOLLOWER - KALMAN IN POLYNOMIAL MODE ONLY")
    print("="*70)
    print("\n📋 STRICT RULE:")
    print("   🔵 Cyan line present  → MEDIAN-BASED (NO Kalman)")
    print("   🟣 Cyan line absent   → POLYNOMIAL-BASED (WITH Kalman)")
    print("\n🎯 POLYNOMIAL MODE FEATURES:")
    print("   • BOTH edges visible → TRUE CENTER (optimal for roundabouts!)")
    print("   • Only right edge    → Fixed offset (consistent with median)")
    print("   • Kalman smoothing   → Reduces oscillations & handles missing data")
    print("\n⚙️  KALMAN PARAMETERS (Polynomial mode only):")
    print("   • Process noise: 0.5")
    print("   • Measurement noise: 5.0 (adaptive)")
    print("   • Max predictions: 5 frames")
    print("\n💡 CONTROLS:")
    print("   • 'q' - Quit")
    print("   • 'k' - Toggle Kalman ON/OFF (polynomial mode only)")
    print("\n🎯 BENEFITS:")
    print("   ✅ Simple & reliable median mode (no filtering needed)")
    print("   ✅ Smooth roundabout handling with Kalman (polynomial mode)")
    print("   ✅ Automatic mode switching based on cyan detection")
    print("="*70 + "\n")
    
    input("Press ENTER to start... ")
    
    follower = HybridLaneFollower()
    follower.run()