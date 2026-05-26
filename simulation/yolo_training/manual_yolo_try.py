import cv2
import numpy as np
import time
import sys
import os
import json
from io import StringIO
from datetime import datetime
from pathlib import Path
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels
from pynput import keyboard

# 🚦 YOLO IMPORT
from ultralytics import YOLO

# Environment objects
from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.yield_sign import QLabsYieldSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

class YOLOTestCollector:
    def __init__(self, model_path=None):
        print("\n" + "="*70)
        print("🚦 YOLO TEST COLLECTOR - Drive & Test Your Model!")
        print("="*70)
        
        # 🤖 LOAD YOLO MODEL (OPTIONAL)
        self.use_yolo = False
        if model_path and os.path.exists(model_path):
            print(f"\n🤖 Loading YOLO model: {model_path}")
            self.yolo_model = YOLO(model_path)
            self.use_yolo = True
            print("✅ YOLO model loaded!")
            
            # Detection config
            self.detection_conf_threshold = 0.05
            self.class_names = {0: 'GREEN', 1: 'RED'}
            self.class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
        else:
            if model_path:
                print(f"\n⚠️  Model not found: {model_path}")
            print("   Running without YOLO detection")
        
        self.qlabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")
        self.qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        
        print("\n🚗 Spawning QCar...")
        self.qcar = QLabsQCar(self.qlabs)
        
        spawn_location = [-0.15, 3, 0.01]
        spawn_rotation = [0, 0, 300]
        
        self.qcar.spawn_id(actorNumber=0, location=spawn_location, rotation=spawn_rotation, waitForConfirmation=True)
        print("✅ QCar spawned!")
        
        # 🚦 Traffic light positions - DEFINE BEFORE spawn_traffic_elements()
        self.light_positions = [
            ("L0", -22.313, 36.363),
            ("L1", -2.95, 5.6),
            ("L2", 6.7, 5.7),
            ("L3", 24.387, 4.74)
        ]
        
        print("\n🚦 Spawning traffic elements...")
        self.spawn_traffic_elements()
        
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        
        # Initialize hardware
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            self.qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            self.use_hardware = True
            print("✅ Hardware control enabled")
        except Exception as e:
            sys.stdin = old_stdin
            self.use_hardware = False
            print("⚠️  No hardware control - simulation only")
        
        # Driving controls
        self.speed = 0.0
        self.steering = 0.0
        self.max_speed = 0.04
        self.max_steering = 0.3
        self.speed_increment = 0.001
        self.steering_increment = 0.01
        
        # Key states
        self.keys_held = {'w': False, 's': False, 'a': False, 'd': False}
        self.action_keys = {'c': False, 'r': False, 'g': False, 'x': False, 'q': False, 'v': False}
        
        # Dataset folders (optional - for capturing images)
        self.save_dir = "traffic_lights_dataset"
        self.red_dir = os.path.join(self.save_dir, "red_lights")
        self.green_dir = os.path.join(self.save_dir, "green_lights")
        
        os.makedirs(self.red_dir, exist_ok=True)
        os.makedirs(self.green_dir, exist_ok=True)
        
        self.red_count = 0
        self.green_count = 0
        
        # Traffic lights - Start with RED (state 0)
        self.current_light_state = 0  # 0=RED, 1=GREEN
        self.set_all_lights(self.current_light_state)
        
        # ROI visualization toggle
        self.show_roi = False
        self.show_distances = True
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        
        print(f"\n{'='*70}")
        if self.use_yolo:
            print(f"🤖 YOLO MODEL READY - Detections will show in real-time!")
            print(f"   Confidence threshold: {self.detection_conf_threshold}")
            print(f"   YOLO uses FULL IMAGE (no ROI cropping)")
            print(f"   🔍 DEBUG MODE: Will print raw model outputs")
        print(f"🚦 Current lights: {self.get_light_state_name(self.current_light_state)}")
        print(f"📏 Distance display: {'ON' if self.show_distances else 'OFF'} (press V to toggle)")
        print(f"{'='*70}")
    
    def spawn_traffic_elements(self):
        # Crosswalks
        walks = [QLabsCrosswalk(self.qlabs) for _ in range(4)]
        walks[0].spawn(location=[-5, 9.5, 0], rotation=[0,0,np.pi/2], scale=[1,1,0.75], configuration=0)
        walks[1].spawn(location=[1.3, 16, 0], rotation=[0,0,0], scale=[1,1,0.75], configuration=0)
        walks[2].spawn(location=[7.7, 9.5, 0], rotation=[0,0,np.pi/2], scale=[1,1,0.75], configuration=0)
        walks[3].spawn(location=[1.3, 3, 0], rotation=[0,0,0], scale=[1,1,0.75], configuration=0)
        
        # Traffic lights
        self.lights = [QLabsTrafficLight(self.qlabs) for _ in range(4)]
        initial_state = 0  # RED
        self.lights[0].spawn(location=[-22.313, 36.363, 0.0], rotation=[0,0,135], configuration=initial_state)
        self.lights[1].spawn(location=[-2.95, 5.6, 0], rotation=[0,0,300], configuration=initial_state)
        self.lights[2].spawn(location=[6.7, 5.7, 0], rotation=[0,0,-np.pi/2], configuration=initial_state)
        self.lights[3].spawn(location=[24.387, 4.74, 0.2], rotation=[0,0,0], configuration=initial_state)
        
        QLabsYieldSign(self.qlabs).spawn(location=[0.4,-13, 0], rotation=[0,0,np.pi])
        
        roundAboutSigns = [QLabsRoundaboutSign(self.qlabs) for _ in range(3)]
        roundAboutSigns[0].spawn(location=[24.5,33, 0], rotation=[0,0,-np.pi/2])
        roundAboutSigns[1].spawn(location=[4.5,40, 0], rotation=[0,0,np.pi])
        roundAboutSigns[2].spawn(location=[10.6,28.5, 0], rotation=[0,0,np.pi])
        
        QLabsStopSign(self.qlabs).spawn(location=[-0.508, -7.327, 0.2], rotation=[0,0, np.pi/2],
                                       scale=[1,1,1], configuration=0, waitForConfirmation=True)
        
        print("   ✅ All traffic elements spawned")
        print("\n📍 Traffic Light Locations:")
        for name, x, y in self.light_positions:
            print(f"   {name}: ({x:.1f}, {y:.1f})")
    
    def set_all_lights(self, state):
        """Set ALL lights - 0=RED, 1=GREEN"""
        for light in self.lights:
            light.set_state(state)
        self.current_light_state = state
        state_name = self.get_light_state_name(state)
        print(f"\n🚦 ALL LIGHTS → {state_name}")
    
    def get_light_state_name(self, state):
        """0=RED, 1=GREEN"""
        return {0: 'RED', 1: 'GREEN'}.get(state, 'UNKNOWN')
    
    def detect_traffic_lights(self, img):
        """Detect traffic lights using YOLO - USES FULL IMAGE - WITH DEBUG"""
        if not self.use_yolo:
            return []
        
        # YOLO uses FULL image (no ROI cropping)
        results = self.yolo_model(img, conf=self.detection_conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                raw_class_id = int(box.cls[0])  # What model ACTUALLY outputs
                confidence = float(box.conf[0])
                
                # 🔍 DEBUG: Print what model outputs
                actual_light = self.get_light_state_name(self.current_light_state)
                print(f"🔍 ACTUAL LIGHT: {actual_light} | MODEL OUTPUT: class_id={raw_class_id} (conf={confidence:.2f})")
                
                # NO FLIP - Use raw output
                class_id = raw_class_id
                
                if class_id in self.class_names:
                    bbox = box.xyxy[0].cpu().numpy()
                    detections.append((class_id, confidence, bbox))
        
        return detections
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels"""
        for class_id, confidence, bbox in detections:
            if class_id not in self.class_names:
                continue
            
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.class_colors[class_id]
            name = self.class_names[class_id]
            
            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Label background
            label = f"{name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Label text
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def show_light_distances(self, img, car_x, car_y):
        """Show distances to all traffic lights"""
        if not self.show_distances:
            return
        
        y_offset = 200
        cv2.putText(img, "Distances to Lights:", (550, y_offset - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Calculate and sort by distance
        distances = []
        for name, lx, ly in self.light_positions:
            dist = np.hypot(car_x - lx, car_y - ly)
            distances.append((name, lx, ly, dist))
        
        distances.sort(key=lambda x: x[3])  # Sort by distance
        
        for i, (name, lx, ly, dist) in enumerate(distances):
            # Color code by distance
            if dist < 10:
                color = (0, 255, 0)  # GREEN - very close
                status = "CLOSE"
            elif dist < 20:
                color = (0, 255, 255)  # YELLOW - medium
                status = "NEAR"
            elif dist < 40:
                color = (0, 165, 255)  # ORANGE - far
                status = "FAR"
            else:
                color = (100, 100, 100)  # GRAY - very far
                status = "VERY FAR"
            
            text = f"{name}: {dist:5.1f}m ({status})"
            if i == 0:  # Highlight closest
                cv2.putText(img, f">>> {text} <<<", (550, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(img, f"    {text}", (550, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
    
    def draw_roi_overlay(self, img):
        """Draw ROI rectangle for visualization"""
        if not self.show_roi:
            return
        
        h, w = img.shape[:2]
        roi_start = int(h * 0.3)  # Top 30%
        roi_end = int(h * 0.8)    # Bottom 80%
        
        # Draw ROI rectangle
        cv2.rectangle(img, (0, roi_start), (w, roi_end), (255, 0, 255), 2)
        cv2.putText(img, "ROI (not used by YOLO)", (10, roi_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = True
                elif key.char in ['c', 'r', 'g', 'x', 'q', 'v']:
                    self.action_keys[key.char] = True
        except AttributeError:
            pass
    
    def on_key_release(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = False
        except AttributeError:
            pass
    
    def apply_controls(self):
        """Apply control inputs"""
        # Speed control
        if self.keys_held['w']:
            self.speed = min(self.speed + self.speed_increment, self.max_speed)
        elif self.keys_held['s']:
            self.speed = max(self.speed - self.speed_increment, -self.max_speed)
        else:
            self.speed *= 0.97
            if abs(self.speed) < 0.001:
                self.speed = 0.0
        
        # Steering control
        if self.keys_held['a']:
            self.steering = min(self.steering + self.steering_increment, self.max_steering)
        elif self.keys_held['d']:
            self.steering = max(self.steering - self.steering_increment, -self.max_steering)
        else:
            self.steering *= 0.9
            if abs(self.steering) < 0.01:
                self.steering = 0.0
    
    def capture_image(self, img, car_x, car_y, car_yaw):
        """Save image to appropriate folder"""
        light_state_name = self.get_light_state_name(self.current_light_state)
        
        if self.current_light_state == 0:  # RED
            self.red_count += 1
            filename = f"red_{self.red_count:04d}.png"
            filepath = os.path.join(self.red_dir, filename)
        else:  # GREEN
            self.green_count += 1
            filename = f"green_{self.green_count:04d}.png"
            filepath = os.path.join(self.green_dir, filename)
        
        cv2.imwrite(filepath, img)
        print(f"📸 Captured: {light_state_name} #{self.red_count if self.current_light_state == 0 else self.green_count}")
    
    def run(self):
        print("\n" + "="*70)
        print("🎮 CONTROLS:")
        print("="*70)
        print("DRIVING:")
        print("   ⬆️  HOLD W = Forward")
        print("   ⬇️  HOLD S = Backward")
        print("   ⬅️  HOLD A = Left")
        print("   ➡️  HOLD D = Right")
        print("\nLIGHTS:")
        print("   R = All lights RED 🔴")
        print("   G = All lights GREEN 🟢")
        print("\nVIEW:")
        print("   V = Toggle distance display")
        print("\nOTHER:")
        print("   C = Capture image (optional)")
        print("   X = Emergency stop")
        print("   Q = Quit")
        
        if self.use_yolo:
            print("\n🤖 YOLO ACTIVE - Watch for bounding boxes!")
            print("   🔍 Watch console for debug output")
        
        print("\n💡 TIP: Drive close to lights (< 10m) for best detection!")
        print("="*70 + "\n")
        
        try:
            while True:
                # Get image
                success_img, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success_img and image_data is not None:
                    img = np.frombuffer(image_data, dtype=np.uint8).reshape((410, 820, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # 🤖 YOLO DETECTION (uses full image)
                    detections = []
                    if self.use_yolo:
                        detections = self.detect_traffic_lights(img)
                    
                    # Get position
                    result = self.qcar.get_world_transform()
                    
                    if result and len(result) == 4 and result[0]:
                        car_x, car_y, car_yaw = result[1][0], result[1][1], result[2][2]
                        
                        # Apply driving controls
                        self.apply_controls()
                        
                        if self.use_hardware:
                            self.qcar_hw.write(self.speed, self.steering)
                        
                        # Handle action keys
                        if self.action_keys['x']:
                            self.speed = 0.0
                            self.steering = 0.0
                            self.action_keys['x'] = False
                            print("🛑 Emergency stop!")
                        
                        if self.action_keys['c']:
                            self.capture_image(img, car_x, car_y, car_yaw)
                            self.action_keys['c'] = False
                        
                        if self.action_keys['r']:
                            self.set_all_lights(0)  # RED
                            self.action_keys['r'] = False
                        
                        if self.action_keys['g']:
                            self.set_all_lights(1)  # GREEN
                            self.action_keys['g'] = False
                        
                        if self.action_keys['v']:
                            self.show_distances = not self.show_distances
                            self.action_keys['v'] = False
                            print(f"📏 Distance display: {'ON' if self.show_distances else 'OFF'}")
                        
                        if self.action_keys['q']:
                            break
                        
                        # 🤖 DRAW YOLO DETECTIONS
                        if self.use_yolo and detections:
                            img = self.draw_detections(img, detections)
                        
                        # Draw ROI overlay (if enabled)
                        self.draw_roi_overlay(img)
                        
                        # Visualization overlay
                        overlay = img.copy()
                        cv2.rectangle(overlay, (0, 0), (820, 180), (0, 0, 0), -1)
                        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                        
                        # Actual light state indicator
                        if self.current_light_state == 0:  # RED
                            light_color = (0, 0, 255)
                            light_name = "RED"
                        else:  # GREEN
                            light_color = (0, 255, 0)
                            light_name = "GREEN"
                        
                        cv2.putText(img, f"ACTUAL LIGHTS: {light_name}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2)
                        
                        # YOLO detection summary
                        if self.use_yolo:
                            cv2.putText(img, f"Image: {img.shape[1]}x{img.shape[0]}", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                            if detections:
                                # Show best detection
                                best = max(detections, key=lambda x: x[1])
                                detected_name = self.class_names[best[0]]
                                detected_conf = best[1]
                                detected_color = self.class_colors[best[0]]
                                
                                cv2.putText(img, f"YOLO SEES: {detected_name} ({detected_conf:.2f})", (10, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, detected_color, 2)
                                
                                # Correctness check
                                if detected_name == light_name:
                                    cv2.putText(img, "✅ CORRECT!", (10, 120),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                else:
                                    cv2.putText(img, "❌ WRONG!", (10, 120),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                cv2.putText(img, f"Total detections: {len(detections)}", (10, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.putText(img, "YOLO SEES: Nothing", (10, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                                cv2.putText(img, "Drive closer to a traffic light!", (10, 120),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Stats
                        cv2.putText(img, f"Position: ({car_x:.1f}, {car_y:.1f})", (10, 175),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Show distances to lights
                        self.show_light_distances(img, car_x, car_y)
                        
                        # Find nearest light
                        nearest = min(self.light_positions, key=lambda x: np.hypot(car_x - x[1], car_y - x[2]))
                        nearest_dist = np.hypot(car_x - nearest[1], car_y - nearest[2])
                        
                        # Navigation hint
                        if nearest_dist > 15:
                            cv2.putText(img, f"🎯 Drive to {nearest[0]} ({nearest_dist:.1f}m)", (10, 400),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.imshow('🤖 YOLO Test Drive - See Your Model in Action!', img)
                
                cv2.waitKey(1)
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if self.use_hardware:
                self.qcar_hw.write(0, 0)
            
            self.listener.stop()
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            print("\n✅ Session complete!")
            if self.red_count > 0 or self.green_count > 0:
                print(f"   Captured: {self.red_count} red, {self.green_count} green\n")

if __name__ == '__main__':
    print("\n🤖 YOLO Test Collector with Distance Display")
    print("="*70)
    
    # 🎯 DEFAULT MODEL PATH
    DEFAULT_MODEL = r"C:\Users\kcksa\Documents\simulation\traffic_light_training\traffic_detector4\weights\best.pt"
    
    # Check if command line argument provided (override default)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📂 Using model from command line: {model_path}")
    else:
        model_path = DEFAULT_MODEL
        print(f"📂 Using default model: {model_path}")
    
    try:
        collector = YOLOTestCollector(model_path)
        collector.run()
    except ImportError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Install: pip install pynput ultralytics\n")