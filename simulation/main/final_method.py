import cv2
import numpy as np
import time
import sys
import os
from io import StringIO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels
from ultralytics import YOLO

# Environment objects
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

class LaneCenterFollower:
    def __init__(self, light_model_path=None, sign_model_path=None):
        print("\n" + "="*70)
        print("LANE CENTERING WITH YOLO DETECTION - Lights & Stop Signs!")
        print("="*70)
        
        # Load YOLO models
        if light_model_path and os.path.exists(light_model_path):
            print(f"\n📦 Loading LIGHT model: {light_model_path}")
            self.light_model = YOLO(light_model_path)
            print("✅ Light model loaded!")
        else:
            self.light_model = None
            print(f"⚠️  No light model loaded - Path: {light_model_path}")
            print(f"⚠️  Exists: {os.path.exists(light_model_path) if light_model_path else 'N/A'}")
        
        if sign_model_path and os.path.exists(sign_model_path):
            print(f"\n📦 Loading SIGN model: {sign_model_path}")
            self.sign_model = YOLO(sign_model_path)
            print("✅ Sign model loaded!")
        else:
            self.sign_model = None
            print(f"⚠️  No sign model loaded - Path: {sign_model_path}")
            print(f"⚠️  Exists: {os.path.exists(sign_model_path) if sign_model_path else 'N/A'}")
        
        # Detection class names
        self.light_class_names = {0: 'RED_LIGHT', 1: 'GREEN_LIGHT'}
        self.light_class_colors = {0: (0, 0, 255), 1: (0, 255, 0)}
        self.sign_class_names = {0: 'STOP'}
        self.sign_class_colors = {0: (0, 0, 255)}
        
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
            location=[-0.15, 1, 0.01],
            rotation=[0, 0, 300],
            waitForConfirmation=True
        )
        print("✅ QCar spawned!")
        
        print("\n🚦 Spawning traffic elements...")
        self.spawn_traffic_elements()
        print("✅ Traffic elements spawned!")
        
        print("\n▶️  Starting simulation...")
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        print("✅ Running!")
        
        # Speed control
        self.base_speed = 0.045
        self.start_time = None
        
        # Slow zone
        self.slow_zone_center = (-20.02, 23.332)
        self.slow_zone_radius = 2.0
        self.slow_zone_speed = 0.035
        self.slow_zone_duration = 3.0
        self.slow_zone_start_time = None
        self.in_slow_zone = False
        self.slow_zone_completed = False
        
        # Waypoint tracking
        self.round16_waypoints = self.load_waypoints('round16_waypoints.npy')
        self.curve12_waypoints = self.load_waypoints('curve12_waypoints.npy')
        self.active_waypoints = None
        self.active_waypoint_name = None
        self.curve12_completed = False
        self.curve12_completion_time = None
        self.post_curve12_slowdown_duration = 5.0
        self.in_special_zone = False
        self.current_waypoint_idx = 0
        self.lookahead = 0.5
        self.waypoint_L = 0.256
        self.special_zone_speed = 0.045
        
        # CURVE12 CONTROL
        self.curve12_enabled = False
        self.first_stop_completed = False
        
        # Cyan detection
        self.cyan_start_threshold = 500
        self.cyan_start_counter = 0
        self.cyan_start_frames = 5
        
        # Control parameters
        self.kp = 0.8
        self.kd = 0.2
        self.previous_error = 0
        
        # ROI
        self.roi_start = 0.55
        self.roi_height = 0.45
        
        # HSV values
        self.lower_cyan = np.array([80, 0, 150])
        self.upper_cyan = np.array([170, 255, 255])
        self.lower_white = np.array([0, 0, 240])
        self.upper_white = np.array([180, 25, 255])
        
        # Fallback
        self.offset_from_white = 240
        self.offset_from_cyan = 250
        self.last_cyan_loss_time = None
        self.cyan_switch_timeout = 3.0
        self.force_white_only = False
        self.was_using_cyan = False
        
        # Smoothing
        self.previous_lane_center = None
        self.previous_strategy = None
        self.transition_frames = 15
        self.transition_counter = 0
        self.transition_start_center = None
        self.transition_target_center = None
        self.ema_alpha = 0.3
        self.ema_lane_center = None
        self.max_steering_change = 0.05
        self.previous_steering = 0.0
        
        # Traffic light positions
        self.light_positions = [
            ("L0", -23.513, 26.363),
            ("L1", -2.152, 3.8),
            ("L2", 26.687, 8.74)
        ]
        
        # TRAFFIC LIGHT SMART DETECTION
        self.light_detection_settings = {
            "L0": {"radius": 12.0, "min_box_size": 10},
            "L1": {"radius": 12.0, "min_box_size": 10},
            "L2": {"radius": 12.0, "min_box_size": 10},
        }
        
        # HORIZONTAL ROI for traffic lights
        self.light_roi_left = 0.2
        self.light_roi_right = 0.8
        
        # Global detection parameters
        self.light_red_confidence = 0.4
        self.light_green_confidence = 0.3
        self.traffic_light_stop_distance = 12.0
        self.at_traffic_light = False
        self.current_traffic_light = None
        self.traffic_stop_start_time = None
        self.traffic_light_timeout = 15.0
        
        # STOP SIGN SMART DETECTION
        self.sign_positions = [
            ("STOP1", -0.508, -7.327),
            ("STOP2", 24.5, 33.0)
        ]
        
        self.sign_detection_radius = 30.0
        self.min_sign_box_size = 500
        
        # STOP SIGN STOP BEHAVIOR
        self.stop_sign_stop_distance = 6.0
        self.stop_sign_stop_duration = 1.0
        self.at_stop_sign = False
        self.stop_sign_start_time = None
        self.stopped_at_signs = set()
        
        # COOLDOWN
        self.last_stopped_sign = None
        self.stop_cooldown_duration = 5.0
        self.stop_cooldown_start_time = None
        
        # TRAFFIC LIGHT AUTOMATIC CYCLING
        self.red_light_duration = 10.0
        self.green_light_duration = 5.0
        self.last_light_cycle_time = time.time()
        self.current_light_state = 1
        
        print(f"\n✅ YOLO Detection: {'Enabled' if (self.light_model and self.sign_model) else 'Disabled'}")
        print(f"   🚦 Traffic Lights: 3 locations")
        print(f"   🛑 Stop Signs: 2 locations")
        print(f"\n🚀 SPEED: Full speed (0.045) from start")
        print(f"\n🗺️  CURVE12: DISABLED at spawn - Will enable after first stop!")
        print(f"\n🎨 CENTERLINE: Dashed Magenta")
    
    def spawn_traffic_elements(self):
        """Spawn traffic lights and stop signs"""
        
        # Crosswalks
        self.walks = [QLabsCrosswalk(self.qlabs) for _ in range(4)]
        self.walks[0].spawn(location=[-5, 9.5, 0], rotation=[0,0,np.pi/2], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[1].spawn(location=[1.3, 16, 0], rotation=[0,0,0], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[2].spawn(location=[7.7, 9.5, 0], rotation=[0,0,np.pi/2], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[3].spawn(location=[1.3, 3, 0], rotation=[0,0,0], 
                           scale=[1,1,0.75], configuration=0)
        
        # Traffic lights - START WITH GREEN
        self.lights = [QLabsTrafficLight(self.qlabs) for _ in range(3)]
        self.lights[0].spawn(location=[-23.513, 26.363, -2.5], rotation=[0,0,135], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[0].set_color(QLabsTrafficLight.COLOR_GREEN)
        self.lights[1].spawn(location=[-2.152, 3.4, -2.5], rotation=[0,0,300], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[1].set_color(QLabsTrafficLight.COLOR_GREEN)
        self.lights[2].spawn(location=[26.687, 8.74, -2.5], rotation=[0,0,0], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[2].set_color(QLabsTrafficLight.COLOR_GREEN)
        
        # Stop signs
        QLabsStopSign(self.qlabs).spawn(location=[-0.508, -7.327, 0.2], 
                                        rotation=[0,0, np.pi/2], scale=[1,1,1], 
                                        configuration=0, waitForConfirmation=True)
        QLabsStopSign(self.qlabs).spawn(location=[24.5, 33, 0.2], 
                                        rotation=[0,0,-np.pi/2], scale=[1,1,1], 
                                        configuration=0, waitForConfirmation=True)
        
        print("   ✅ Crosswalks: 4")
        print("   ✅ Traffic Lights: 3 (Starting GREEN)")
        print("   ✅ Stop Signs: 2")
    
    def cycle_traffic_lights(self):
        """Cycle traffic lights automatically"""
        current_time = time.time()
        elapsed = current_time - self.last_light_cycle_time
        
        if self.current_light_state == 0:
            cycle_duration = self.red_light_duration
        else:
            cycle_duration = self.green_light_duration
        
        if elapsed >= cycle_duration:
            if self.current_light_state == 0:
                self.current_light_state = 1
                light_color = QLabsTrafficLight.COLOR_GREEN
                print(f"🚦 Traffic lights: RED → GREEN")
            else:
                self.current_light_state = 0
                light_color = QLabsTrafficLight.COLOR_RED
                print(f"🚦 Traffic lights: GREEN → RED")
            
            for light in self.lights:
                light.set_color(light_color)
            
            self.last_light_cycle_time = current_time
    
    def should_detect_lights(self, car_x, car_y):
        """Check if car is close enough to any traffic light"""
        closest_light = None
        closest_distance = float('inf')
        
        for name, light_x, light_y in self.light_positions:
            distance = np.hypot(car_x - light_x, car_y - light_y)
            
            light_settings = self.light_detection_settings.get(name, {"radius": 12.0, "min_box_size": 300})
            detection_radius = light_settings["radius"]
            
            if distance <= detection_radius:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_light = (name, distance, light_settings["min_box_size"])
        
        if closest_light:
            return True, closest_light[0], closest_light[1], closest_light[2]
        
        return False, None, None, None
    
    def detect_lights_smart(self, img, car_x, car_y, conf_threshold=0.3):
        """SMART TRAFFIC LIGHT DETECTION with HORIZONTAL ROI"""
        
        should_detect, nearest_light, distance, min_box_size = self.should_detect_lights(car_x, car_y)
        
        if not should_detect:
            return []
        
        if self.light_model is None:
            return []
        
        # HORIZONTAL ROI
        h, w = img.shape[:2]
        roi_start_x = int(w * self.light_roi_left)
        roi_end_x = int(w * self.light_roi_right)
        roi_img = img[:, roi_start_x:roi_end_x, :]
        
        results = self.light_model(roi_img, conf=conf_threshold, verbose=False)
        detections = []
        filtered_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = bbox
                x1 += roi_start_x
                x2 += roi_start_x
                bbox = np.array([x1, y1, x2, y2])
                
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                if box_area < min_box_size:
                    filtered_count += 1
                    continue
                
                if class_id == 0:
                    if confidence < self.light_red_confidence:
                        filtered_count += 1
                        continue
                elif class_id == 1:
                    if confidence < self.light_green_confidence:
                        filtered_count += 1
                        continue
                
                detections.append((class_id, confidence, bbox))
        
        if filtered_count > 0:
            print(f"🔍 Filtered {filtered_count} side/small light detections")
        
        return detections
    
    def check_traffic_light_proximity(self, car_x, car_y, light_detections):
        """Check if car should stop at RED traffic light"""
        
        red_light_detected = False
        red_count = 0
        green_count = 0
        
        for class_id, confidence, bbox in light_detections:
            if class_id == 0:
                red_light_detected = True
                red_count += 1
            elif class_id == 1:
                green_count += 1
        
        if red_count > 0 or green_count > 0:
            print(f"🔍 YOLO: RED={red_count}, GREEN={green_count} | Car: ({car_x:.1f}, {car_y:.1f})")
        
        if self.at_traffic_light:
            if self.traffic_stop_start_time is None:
                self.traffic_stop_start_time = time.time()
            
            stopped_duration = time.time() - self.traffic_stop_start_time
            
            if stopped_duration > self.traffic_light_timeout:
                print(f"\n⚠️  TIMEOUT at {self.current_traffic_light}")
                self.at_traffic_light = False
                self.current_traffic_light = None
                self.traffic_stop_start_time = None
                return False, None, None
            
            if red_light_detected:
                print(f"🔍 STAYING STOPPED at {self.current_traffic_light}")
                return True, self.current_traffic_light, "RED"
            else:
                print(f"\n✅ RESUMING from {self.current_traffic_light}")
                self.at_traffic_light = False
                self.current_traffic_light = None
                self.traffic_stop_start_time = None
                return False, None, None
        
        if not red_light_detected:
            return False, None, None
        
        nearest_light = None
        nearest_distance = float('inf')
        nearest_light_pos = None
        
        for light_name, light_x, light_y in self.light_positions:
            distance = np.hypot(car_x - light_x, car_y - light_y)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_light = light_name
                nearest_light_pos = (light_x, light_y)
        
        print(f"🔍 Nearest: {nearest_light} at {nearest_distance:.2f}m")
        
        if nearest_distance <= self.traffic_light_stop_distance:
            light_x, light_y = nearest_light_pos
            is_approaching = False
            
            if nearest_light == "L0":
                if car_y < light_y - 2:
                    is_approaching = True
                elif car_y > light_y + 2:
                    is_approaching = True
            elif nearest_light == "L1":
                is_approaching = (car_y > light_y)
            elif nearest_light == "L2":
                is_approaching = (car_x < light_x)
            
            if is_approaching:
                self.at_traffic_light = True
                self.current_traffic_light = nearest_light
                self.traffic_stop_start_time = time.time()
                print(f"\n🚦 STOPPING at {nearest_light} - RED")
                return True, nearest_light, "RED"
        
        return False, None, None
    
    def should_detect_signs(self, car_x, car_y):
        """Check if car is close enough to any stop sign"""
        for name, sign_x, sign_y in self.sign_positions:
            distance = np.hypot(car_x - sign_x, car_y - sign_y)
            if distance <= self.sign_detection_radius:
                return True, name, distance
        return False, None, None
    
    def check_stop_sign_proximity(self, car_x, car_y, sign_detections):
        """Check if car should stop at stop sign"""
        if self.stop_cooldown_start_time is not None:
            cooldown_elapsed = time.time() - self.stop_cooldown_start_time
            if cooldown_elapsed < self.stop_cooldown_duration:
                return False, None, 0.0
            else:
                self.stop_cooldown_start_time = None
                self.last_stopped_sign = None
        
        if not sign_detections:
            if self.at_stop_sign:
                self.at_stop_sign = False
            return False, None, 0.0
        
        detected_signs = []
        for name, sign_x, sign_y in self.sign_positions:
            distance = np.hypot(car_x - sign_x, car_y - sign_y)
            if distance <= self.sign_detection_radius:
                detected_signs.append((name, sign_x, sign_y, distance))
        
        if not detected_signs:
            if self.at_stop_sign:
                self.at_stop_sign = False
            return False, None, 0.0
        
        detected_signs.sort(key=lambda x: x[3])
        nearest_sign, sign_x, sign_y, nearest_distance = detected_signs[0]
        
        if nearest_distance <= self.stop_sign_stop_distance:
            if not self.at_stop_sign:
                self.at_stop_sign = True
                self.stop_sign_start_time = time.time()
                print(f"\n🛑 STOPPING at {nearest_sign} ({nearest_distance:.2f}m)")
            
            elapsed = time.time() - self.stop_sign_start_time
            remaining = max(0, self.stop_sign_stop_duration - elapsed)
            
            if elapsed >= self.stop_sign_stop_duration:
                if self.at_stop_sign:
                    print(f"\n✅ RESUMING from {nearest_sign}")
                    self.at_stop_sign = False
                    self.last_stopped_sign = nearest_sign
                    self.stop_cooldown_start_time = time.time()
                return False, nearest_sign, 0.0
            
            return True, nearest_sign, remaining
        else:
            if self.at_stop_sign:
                self.at_stop_sign = False
            return False, None, 0.0
    
    def detect_signs_smart(self, img, car_x, car_y, conf_threshold=0.5):
        """SMART STOP SIGN DETECTION"""
        
        should_detect, nearest_sign, distance = self.should_detect_signs(car_x, car_y)
        
        if not should_detect:
            return []
        
        if self.sign_model is None:
            return []
        
        results = self.sign_model(img, conf=conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = bbox
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                if box_area >= self.min_sign_box_size:
                    detections.append((class_id, confidence, bbox))
        
        return detections
    
    def draw_detections(self, img, light_detections, sign_detections):
        """Draw YOLO detections with labels on RIGHT side"""
        # Draw light detections
        for class_id, confidence, bbox in light_detections:
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.light_class_colors.get(class_id, (255, 255, 255))
            name = self.light_class_names.get(class_id, 'UNKNOWN')
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {confidence:.2f}"
            cv2.putText(img, label, (x2+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        
        # Draw sign detections
        for class_id, confidence, bbox in sign_detections:
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.sign_class_colors.get(class_id, (255, 255, 255))
            name = self.sign_class_names.get(class_id, 'UNKNOWN')
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {confidence:.2f}"
            cv2.putText(img, label, (x2+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        
        return img
    
    def draw_dashed_centerline(self, img, color=(255, 0, 255), dash_length=20, thickness=2):
        """Draw DASHED MAGENTA centerline"""
        h, w = img.shape[:2]
        center_x = w // 2
        
        for y in range(0, h, dash_length * 2):
            cv2.line(img, (center_x, y), (center_x, min(y + dash_length, h)), color, thickness)
        
        return img
    
    def load_waypoints(self, filename):
        """Load waypoints"""
        if os.path.exists(filename):
            try:
                waypoints = np.load(filename)
                print(f"✅ Loaded: {filename} ({len(waypoints)} points)")
                return waypoints
            except Exception as e:
                print(f"⚠️  Error: {e}")
                return None
        else:
            print(f"ℹ️  Not found: {filename}")
            return None
    
    def smooth_lane_center_transition(self, new_lane_center, new_strategy):
        """Smooth transition"""
        if new_lane_center is None:
            return None
        
        strategy_changed = (self.previous_strategy is not None and 
                          self.previous_strategy != new_strategy)
        
        if strategy_changed:
            if self.transition_counter == 0:
                self.transition_start_center = self.previous_lane_center
                self.transition_target_center = new_lane_center
                self.transition_counter = 1
            
            if self.transition_counter < self.transition_frames:
                blend_ratio = self.transition_counter / self.transition_frames
                
                if self.transition_start_center and self.transition_target_center:
                    blended_x = int(self.transition_start_center[0] * (1 - blend_ratio) + 
                                  self.transition_target_center[0] * blend_ratio)
                    blended_y = int(self.transition_start_center[1] * (1 - blend_ratio) + 
                                  self.transition_target_center[1] * blend_ratio)
                    blended_center = (blended_x, blended_y)
                    
                    self.transition_counter += 1
                    return blended_center
                else:
                    self.transition_counter = self.transition_frames
                    return new_lane_center
            else:
                self.transition_counter = 0
                self.transition_start_center = None
                self.transition_target_center = None
                return new_lane_center
        else:
            self.transition_counter = 0
            self.transition_start_center = None
            self.transition_target_center = None
            return new_lane_center
    
    def apply_ema_filter(self, new_lane_center):
        """EMA filter"""
        if new_lane_center is None:
            return self.ema_lane_center
        
        if self.ema_lane_center is None:
            self.ema_lane_center = new_lane_center
            return self.ema_lane_center
        
        smoothed_x = int(self.ema_alpha * new_lane_center[0] + 
                        (1 - self.ema_alpha) * self.ema_lane_center[0])
        smoothed_y = int(self.ema_alpha * new_lane_center[1] + 
                        (1 - self.ema_alpha) * self.ema_lane_center[1])
        
        self.ema_lane_center = (smoothed_x, smoothed_y)
        return self.ema_lane_center
    
    def apply_steering_rate_limit(self, new_steering):
        """Steering rate limit"""
        steering_change = new_steering - self.previous_steering
        
        if abs(steering_change) > self.max_steering_change:
            limited_steering = self.previous_steering + np.sign(steering_change) * self.max_steering_change
            self.previous_steering = limited_steering
            return limited_steering
        else:
            self.previous_steering = new_steering
            return new_steering
    
    def check_timed_slow_zone(self, car_x, car_y):
        """Check slow zone"""
        distance = np.hypot(car_x - self.slow_zone_center[0], car_y - self.slow_zone_center[1])
        
        if self.slow_zone_completed:
            return False, 0.0
        
        if distance <= self.slow_zone_radius:
            if not self.in_slow_zone:
                self.in_slow_zone = True
                self.slow_zone_start_time = time.time()
                print(f"\n🐌 SLOW ZONE ({car_x:.1f}, {car_y:.1f})")
            
            elapsed = time.time() - self.slow_zone_start_time
            remaining = max(0, self.slow_zone_duration - elapsed)
            
            if elapsed >= self.slow_zone_duration:
                self.in_slow_zone = False
                self.slow_zone_completed = True
                print(f"\n✅ SLOW ZONE DONE")
                return False, 0.0
            
            return True, remaining
        else:
            if self.in_slow_zone:
                self.in_slow_zone = False
                self.slow_zone_completed = True
            return False, 0.0
    
    def check_special_zone(self, car_x, car_y, at_stop_sign=False):
        """Check waypoint zones"""
        
        if at_stop_sign and not self.in_special_zone:
            return False
        
        if self.curve12_waypoints is not None and not self.in_special_zone and not self.curve12_completed and self.curve12_enabled:
            curve12_start = self.curve12_waypoints[0]
            distance_to_curve12_start = np.hypot(car_x - curve12_start[0], car_y - curve12_start[1])
            curve12_entry_threshold = 5.0
            
            if distance_to_curve12_start <= curve12_entry_threshold:
                print(f"\n🗺️  CURVE12 ({car_x:.1f}, {car_y:.1f})")
                self.in_special_zone = True
                self.active_waypoints = self.curve12_waypoints
                self.active_waypoint_name = "CURVE12"
                self.current_waypoint_idx = 0
                self.cyan_start_counter = 0
                self.ema_lane_center = None
                self.previous_steering = 0.0
                return True
        
        if self.round16_waypoints is not None:
            round16_box = (12.5, 20.0, 23.0, 45.5)
            min_x, min_y, max_x, max_y = round16_box
            
            if min_x <= car_x <= max_x and min_y <= car_y <= max_y:
                if not self.in_special_zone:
                    print(f"\n🗺️  ROUND16 ({car_x:.1f}, {car_y:.1f})")
                    self.in_special_zone = True
                    self.active_waypoints = self.round16_waypoints
                    self.active_waypoint_name = "ROUND16"
                    self.current_waypoint_idx = 0
                    self.cyan_start_counter = 0
                    self.ema_lane_center = None
                    self.previous_steering = 0.0
                return True
        
        if self.in_special_zone:
            if self.active_waypoint_name == "CURVE12":
                if self.current_waypoint_idx >= 104:
                    print(f"\n🔄 CURVE12 DONE")
                    self.in_special_zone = False
                    self.active_waypoints = None
                    self.active_waypoint_name = None
                    self.curve12_completed = True
                    self.curve12_completion_time = time.time()
                    return False
                return True
            
            elif self.active_waypoint_name == "ROUND16":
                round16_box = (12.5, 20.0, 23.0, 45.5)
                min_x, min_y, max_x, max_y = round16_box
                if min_x <= car_x <= max_x and min_y <= car_y <= max_y:
                    return True
            
            print(f"\n👁️  EXITED {self.active_waypoint_name}")
            self.in_special_zone = False
            self.active_waypoints = None
            self.active_waypoint_name = None
            self.current_waypoint_idx = 0
            self.cyan_start_counter = 0
            self.force_white_only = False
            self.last_cyan_loss_time = None
            self.was_using_cyan = False
        
        return False
    
    def force_exit_waypoint_mode(self):
        """Exit waypoint mode"""
        print(f"\n🚦 CYAN LINE - Exiting waypoint")
        self.in_special_zone = False
        self.current_waypoint_idx = 0
        self.cyan_start_counter = 0
        self.force_white_only = False
        self.last_cyan_loss_time = None
        self.was_using_cyan = False
    
    def find_target_waypoint(self, car_x, car_y):
        """Find target waypoint"""
        waypoints = self.active_waypoints
        if waypoints is None:
            return None, None, 0
        
        closest_dist = float('inf')
        closest_idx = self.current_waypoint_idx
        
        for i in range(self.current_waypoint_idx, len(waypoints)):
            wx, wy = waypoints[i]
            dist = np.hypot(wx - car_x, wy - car_y)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        for i in range(closest_idx, len(waypoints)):
            wx, wy = waypoints[i]
            dist = np.hypot(wx - car_x, wy - car_y)
            if dist >= self.lookahead:
                self.current_waypoint_idx = i
                return wx, wy, i
        
        self.current_waypoint_idx = len(waypoints) - 1
        return waypoints[-1][0], waypoints[-1][1], len(waypoints) - 1
    
    def pure_pursuit_steering(self, car_x, car_y, car_yaw, target_x, target_y):
        """Pure Pursuit"""
        dx = target_x - car_x
        dy = target_y - car_y
        target_angle = np.arctan2(dy, dx)
        alpha = target_angle - car_yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        Ld = np.hypot(dx, dy)
        
        if Ld < 0.01:
            return 0.0
        
        steering = np.arctan2(2 * self.waypoint_L * np.sin(alpha), Ld)
        return np.clip(steering, -0.3, 0.3)
    
    def detect_lane_boundaries(self, frame):
        """Detect lanes"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        kernel = np.ones((5,5), np.uint8)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        h, w = frame.shape[:2]
        roi_start_px = int(h * self.roi_start)
        roi_end_px = int(h * min(self.roi_start + self.roi_height, 1.0))
        
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
        
        current_time = time.time()
        
        if self.was_using_cyan and not cyan_center:
            if self.last_cyan_loss_time is None:
                self.last_cyan_loss_time = current_time
            elif current_time - self.last_cyan_loss_time <= self.cyan_switch_timeout:
                if not self.force_white_only:
                    self.force_white_only = True
        
        if cyan_center and not self.force_white_only:
            self.last_cyan_loss_time = None
            self.was_using_cyan = True
        
        lane_center = None
        strategy = "NONE"
        
        if self.force_white_only and white_center:
            target_x = white_center[0] - self.offset_from_white
            target_y = white_center[1]
            lane_center = (target_x, target_y)
            strategy = "FORCED WHITE"
        elif cyan_center and white_center:
            center_x = (cyan_center[0] + white_center[0]) // 2
            center_y = (cyan_center[1] + white_center[1]) // 2
            lane_center = (center_x, center_y)
            strategy = "BOTH"
            self.was_using_cyan = True
        elif white_center:
            target_x = white_center[0] - self.offset_from_white
            target_y = white_center[1]
            lane_center = (target_x, target_y)
            strategy = "WHITE"
            self.was_using_cyan = False
        elif cyan_center:
            target_x = cyan_center[0] + self.offset_from_cyan
            target_y = cyan_center[1]
            lane_center = (target_x, target_y)
            strategy = "CYAN"
            self.was_using_cyan = True
        
        cyan_pixels = cv2.countNonZero(cyan_roi)
        white_pixels = cv2.countNonZero(white_roi)
        combined_mask = cv2.bitwise_or(cyan_mask, white_mask)
        
        return lane_center, cyan_center, white_center, combined_mask, cyan_pixels, white_pixels, (roi_start_px, roi_end_px), strategy
    
    def calculate_steering(self, lane_center, width):
        """Calculate steering"""
        if lane_center is None:
            self.previous_error = 0
            return 0.0
        
        center_x = width // 2
        error = (lane_center[0] - center_x) / center_x
        
        derivative = error - self.previous_error
        steering = -self.kp * error - self.kd * derivative
        self.previous_error = error
        
        steering = self.apply_steering_rate_limit(steering)
        
        return np.clip(steering, -0.3, 0.3)
    
    def run(self):
        """Main loop"""
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            use_hardware = True
            print("✅ Hardware enabled")
        except:
            sys.stdin = old_stdin
            use_hardware = False
            print("⚠️  No hardware")
        
        print("\n🚗 Starting...\n")
        
        self.start_time = time.time()
        frame_count = 0
        detected_count = 0
        lost_count = 0
        
        try:
            while True:
                frame_count += 1
                
                self.cycle_traffic_lights()
                
                result = self.qcar.get_world_transform()
                if result and len(result) == 4:
                    success_pos = result[0]
                    position = result[1]
                    orientation = result[2]
                    
                    if success_pos and position and orientation:
                        car_x = position[0]
                        car_y = position[1]
                        car_yaw = orientation[2]
                    else:
                        car_x, car_y, car_yaw = 0, 0, 0
                else:
                    car_x, car_y, car_yaw = 0, 0, 0
                
                in_timed_slow, time_remaining = self.check_timed_slow_zone(car_x, car_y)
                current_base_speed = self.base_speed
                
                success, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success and image_data is not None:
                    try:
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        img = img.reshape((410, 820, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        light_detections = self.detect_lights_smart(img, car_x, car_y, conf_threshold=0.3)
                        sign_detections = self.detect_signs_smart(img, car_x, car_y, conf_threshold=0.8)
                        
                        at_stop_sign, stop_sign_name, stop_time_remaining = self.check_stop_sign_proximity(car_x, car_y, sign_detections)
                        at_red_light, traffic_light_name, light_color = self.check_traffic_light_proximity(car_x, car_y, light_detections)
                        
                        in_special_zone = self.check_special_zone(car_x, car_y, at_stop_sign)
                        
                        if in_special_zone:
                            target_x, target_y, wp_idx = self.find_target_waypoint(car_x, car_y)
                            steering = self.pure_pursuit_steering(car_x, car_y, car_yaw, target_x, target_y)
                            
                            result = self.detect_lane_boundaries(img)
                            lane_center_vision, cyan_pos, white_pos, mask, cyan_px, white_px, roi_bounds, strategy = result
                            
                            if self.active_waypoint_name == "CURVE12":
                                mode = "WAYPOINT"
                                strategy = f"{self.active_waypoint_name} [STRICT] [{wp_idx+1}/{len(self.active_waypoints)}]"
                                lane_center = (410, 300)
                                mode_color = (255, 0, 255)
                                current_base_speed = self.special_zone_speed
                            elif cyan_px > self.cyan_start_threshold:
                                self.cyan_start_counter += 1
                                if self.cyan_start_counter >= self.cyan_start_frames:
                                    self.force_exit_waypoint_mode()
                                    in_special_zone = False
                                    lane_center = lane_center_vision
                                    steering = self.calculate_steering(lane_center, img.shape[1])
                                    mode = "VISION"
                                    mode_color = (0, 255, 0)
                                else:
                                    mode = "WAYPOINT"
                                    strategy = f"{self.active_waypoint_name} [{wp_idx+1}/{len(self.active_waypoints)}]"
                                    lane_center = (410, 300)
                                    mode_color = (0, 255, 255)
                                    current_base_speed = self.special_zone_speed
                            else:
                                self.cyan_start_counter = 0
                                mode = "WAYPOINT"
                                strategy = f"{self.active_waypoint_name} [{wp_idx+1}/{len(self.active_waypoints)}]"
                                lane_center = (410, 300)
                                mode_color = (255, 165, 0)
                                current_base_speed = self.special_zone_speed
                        else:
                            result = self.detect_lane_boundaries(img)
                            lane_center_raw, cyan_pos, white_pos, mask, cyan_px, white_px, roi_bounds, strategy = result
                            
                            lane_center_blended = self.smooth_lane_center_transition(lane_center_raw, strategy)
                            lane_center = self.apply_ema_filter(lane_center_blended)
                            steering = self.calculate_steering(lane_center, img.shape[1])
                            
                            mode = "VISION"
                            mode_color = (0, 255, 0)
                            
                            self.previous_lane_center = lane_center_raw
                            self.previous_strategy = strategy
                        
                        # Speed control
                        in_post_curve12_slowdown = False
                        if self.curve12_completed and self.curve12_completion_time is not None:
                            elapsed = time.time() - self.curve12_completion_time
                            if elapsed < self.post_curve12_slowdown_duration:
                                in_post_curve12_slowdown = True
                            else:
                                self.curve12_completion_time = None
                                self.curve12_completed = False
                        
                        if at_red_light:
                            detected_count += 1
                            lost_count = 0
                            current_speed = 0.0
                            speed_reason = f"RED ({traffic_light_name})"
                        elif at_stop_sign:
                            detected_count += 1
                            lost_count = 0
                            current_speed = 0.0
                            speed_reason = f"STOP ({stop_time_remaining:.1f}s)"
                            
                            if not self.first_stop_completed and stop_time_remaining < 0.5:
                                self.first_stop_completed = True
                                self.curve12_enabled = True
                                print(f"\n✅ CURVE12 ENABLED")
                        elif lane_center or in_special_zone:
                            detected_count += 1
                            lost_count = 0
                            
                            if self.active_waypoint_name == "CURVE12":
                                current_speed = self.special_zone_speed
                                speed_reason = "CURVE12"
                            elif in_post_curve12_slowdown:
                                current_speed = 0.035
                                speed_reason = "POST-C12"
                            elif in_timed_slow:
                                current_speed = self.slow_zone_speed
                                speed_reason = "SLOW"
                            else:
                                current_speed = current_base_speed
                                speed_reason = "NORMAL"
                        else:
                            lost_count += 1
                            if lost_count < 20:
                                current_speed = current_base_speed * 0.5
                                speed_reason = "SEARCH"
                            else:
                                current_speed = 0
                                speed_reason = "STOP"
                            
                            if lost_count > 100:
                                print(f"\n⚠️  Lost lane!")
                                break
                        
                        if use_hardware:
                            qcar_hw.write(current_speed, steering)
                        
                        # ✅ VISUALIZATION - Green ROI + Dashed Magenta Centerline
                        vis = img.copy()
                        h, w = vis.shape[:2]
                        
                        # Draw YOLO detections
                        vis = self.draw_detections(vis, light_detections, sign_detections)
                        
                        # ✅ Draw DASHED YELLOW CENTERLINE
                        vis = self.draw_dashed_centerline(vis, color=(0,255,255), dash_length=20, thickness=1)
                        
                        # MODE indicator
                        cv2.putText(vis, f"MODE: {mode}", (10, h-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        
                        if mode == "VISION":
                            roi_start_px, roi_end_px = roi_bounds
                            roi_color = (255, 255, 255)
                            
                            # ✅ Draw GREEN ROI RECTANGLE (kept as is)
                            cv2.rectangle(vis, (0, roi_start_px), (w, roi_end_px), roi_color, 1)
                            
                            if cyan_pos:
                                cv2.circle(vis, cyan_pos, 10, (255, 255, 0), -1)
                                cv2.putText(vis, "CYAN", (cyan_pos[0]-20, cyan_pos[1]-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            if white_pos:
                                cv2.circle(vis, white_pos, 10, (255, 255, 255), -1)
                                cv2.putText(vis, "WHITE", (white_pos[0]-25, white_pos[1]-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            if lane_center:
                                cv2.circle(vis, lane_center, 15, (0, 255, 0), -1)
                                cv2.putText(vis, "TARGET", (lane_center[0]-30, lane_center[1]+35),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.line(vis, (w//2, lane_center[1]), lane_center, (0, 255, 0), 3)
                        
                        # Status text - BOTTOM
                        cv2.putText(vis, f"Pos: ({car_x:.1f}, {car_y:.1f})", (10, h-80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(vis, f"Speed: {current_speed:.3f} ({speed_reason})", (10, h-60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        cv2.putText(vis, f"Steering: {steering:.2f}", (10, h-40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        
                        # Detection count
                        if light_detections or sign_detections:
                            cv2.putText(vis, f"Detect: L:{len(light_detections)} S:{len(sign_detections)}", 
                                       (w-180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        #cv2.imshow('Lane Follower', vis)
                        # Resize to 3/4 size for smaller window
                        vis_resized = cv2.resize(vis, (615, 308))
                        cv2.imshow('Lane Follower', vis_resized)
                        
                    except Exception as e:
                        print(f"⚠️  Error: {e}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
        
        finally:
            if use_hardware:
                qcar_hw.write(0, 0)
            
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            if frame_count > 0:
                print(f"\n📊 Stats:")
                print(f"   Frames: {frame_count}")
                print(f"   Detected: {detected_count} ({(detected_count/frame_count)*100:.1f}%)")
            
            print("\n✅ Done!\n")

if __name__ == '__main__':
    light_model = r"C:\Users\kcksa\Documents\simulation\traffic_light_training\traffic_light_detector\weights\best.pt"
    sign_model = r"C:\Users\kcksa\Documents\simulation\traffic_sign_training\traffic_sign_detector\weights\best.pt"
    
    follower = LaneCenterFollower(light_model, sign_model)
    follower.run()