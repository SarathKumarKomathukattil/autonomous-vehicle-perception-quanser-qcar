"""
Combined Traffic Light & Stop Sign Detector Test with Manual Driving
CLEAN VERSION - Only Lights and Stop Signs
"""

import cv2
import numpy as np
from ultralytics import YOLO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels
import time
import sys
import os
from io import StringIO
from pynput import keyboard

# Environment objects
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

class CombinedDetector:
    def __init__(self, light_model_path, sign_model_path):
        print("\n" + "="*70)
        print("🚦🛑 COMBINED DETECTOR - Lights & Stop Signs")
        print("="*70)
        
        # Load traffic light model
        print(f"\n📦 Loading LIGHT model: {light_model_path}")
        self.light_model = YOLO(light_model_path)
        print("✅ Light model loaded!")
        
        # Load traffic sign model
        print(f"\n📦 Loading SIGN model: {sign_model_path}")
        self.sign_model = YOLO(sign_model_path)
        print("✅ Sign model loaded!")
        
        # Light class names
        self.light_class_names = {
            0: 'RED_LIGHT',
            1: 'GREEN_LIGHT'
        }
        
        self.light_class_colors = {
            0: (0, 0, 255),    # Red
            1: (0, 255, 0)     # Green
        }
        
        # Sign class names
        self.sign_class_names = {
            0: 'STOP'
        }
        
        self.sign_class_colors = {
            0: (0, 0, 255)  # Red
        }
        
        # Detection stats
        self.light_detection_count = {'RED_LIGHT': 0, 'GREEN_LIGHT': 0}
        self.sign_detection_count = {'STOP': 0}
        
        # Manual driving controls
        self.speed = 0.0
        self.steering = 0.0
        self.max_speed = 0.04
        self.max_steering = 0.3
        self.speed_increment = 0.001
        self.steering_increment = 0.01
        
        # Key states
        self.keys_held = {'w': False, 's': False, 'a': False, 'd': False}
        self.action_keys = {'r': False, 'g': False, 'x': False, 'q': False}
        
        # Traffic light positions
        self.light_positions = [
            ("L0", -23.513, 26.363),
            ("L1", -2.152, 3.8),
            ("L3", 26.687, 8.74)
        ]
        
        # Sign positions
        self.sign_positions = [
            ("STOP1", -0.508, -7.327),
            ("STOP2", 24.5, 33.0)
        ]
        
        # Current light state
        self.current_light_state = 0  # 0=RED, 1=GREEN
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        
    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = True
                elif key.char in ['r', 'g', 'x', 'q']:
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
        """Apply manual driving controls"""
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
    
    def spawn_traffic_elements(self, qlabs):
        """Spawn traffic lights and stop signs"""
        # Crosswalks
        walks = [QLabsCrosswalk(qlabs) for _ in range(4)]
        walks[0].spawn(location=[-5, 9.5, 0], rotation=[0,0,np.pi/2], scale=[1,1,0.75], configuration=0)
        walks[1].spawn(location=[1.3, 16, 0], rotation=[0,0,0], scale=[1,1,0.75], configuration=0)
        walks[2].spawn(location=[7.7, 9.5, 0], rotation=[0,0,np.pi/2], scale=[1,1,0.75], configuration=0)
        walks[3].spawn(location=[1.3, 3, 0], rotation=[0,0,0], scale=[1,1,0.75], configuration=0)
        
        # Traffic lights
        self.lights = [QLabsTrafficLight(qlabs) for _ in range(3)]
        self.lights[0].spawn(location=[-23.513, 26.363, -2.5], rotation=[0,0,135], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[1].spawn(location=[-2.152, 3.8, -2.5], rotation=[0,0,300], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[2].spawn(location=[26.687, 8.74, -2.5], rotation=[0,0,0], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        
        # Stop signs
        QLabsStopSign(qlabs).spawn(location=[-0.508, -7.327, 0.2], rotation=[0,0, np.pi/2],
                                   scale=[1,1,1], configuration=0, waitForConfirmation=True)
        
        QLabsStopSign(qlabs).spawn(location=[24.5, 33, 0.2], rotation=[0,0,-np.pi/2],
                                   scale=[1,1,1], configuration=0, waitForConfirmation=True)
        
        print("   ✅ All traffic elements spawned")
        print("   Traffic lights: 3 lights (lowered 2.5m, scaled 1.8x)")
        print("   Stop signs: 2 signs")
        print("\n   Traffic Light Locations:")
        print("      L0: (-23.5, 26.4, -2.5)")
        print("      L1: (-2.15, 3.8, -2.5)")
        print("      L3: (26.7, 8.74, -2.5)")
        print("\n   Stop Sign Locations:")
        print("      STOP1: (-0.508, -7.327)")
        print("      STOP2: (24.5, 33.0)")
    
    def set_all_lights(self, state):
        """Set all traffic lights"""
        for light in self.lights:
            light.set_state(state)
        self.current_light_state = state
        state_name = "RED" if state == 0 else "GREEN"
        print(f"\n🚦 ALL LIGHTS → {state_name}")
        
    def detect_lights(self, img, conf_threshold=0.3):
        """Run traffic light detection"""
        results = self.light_model(img, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                detections.append(('light', class_id, confidence, bbox))
        
        return detections
    
    def detect_signs(self, img, conf_threshold=0.3):
        """Run traffic sign detection"""
        results = self.sign_model(img, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                detections.append(('sign', class_id, confidence, bbox))
        
        return detections
    
    def draw_detections(self, img, light_detections, sign_detections):
        """Draw all detections on image"""
        # Draw light detections
        for det_type, class_id, confidence, bbox in light_detections:
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.light_class_colors.get(class_id, (255, 255, 255))
            name = self.light_class_names.get(class_id, 'UNKNOWN')
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label = f"{name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw sign detections
        for det_type, class_id, confidence, bbox in sign_detections:
            x1, y1, x2, y2 = bbox.astype(int)
            color = self.sign_class_colors.get(class_id, (255, 255, 255))
            name = self.sign_class_names.get(class_id, 'UNKNOWN')
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label = f"{name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img
    
    def run_detection_test(self):
        """Run combined detection on QCar camera feed"""
        print("\n🚗 Setting up QCar...")
        
        # Connect to QLabs
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        
        # Spawn QCar
        qcar = QLabsQCar(qlabs)
        qcar.spawn_id(actorNumber=0, location=[-0.15, 3, 0.01], 
                     rotation=[0, 0, 300], waitForConfirmation=True)
        
        # Spawn traffic elements
        print("🚦 Spawning traffic elements...")
        self.spawn_traffic_elements(qlabs)
        
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        
        # Initialize hardware
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            use_hardware = True
            print("✅ Hardware control enabled")
        except:
            sys.stdin = old_stdin
            use_hardware = False
            print("⚠️  No hardware control - simulation only")
        
        print("✅ QCar ready!")
        print("\n" + "="*70)
        print("🎮 CONTROLS:")
        print("="*70)
        print("   W/A/S/D = Drive manually")
        print("   R = Set all lights to RED")
        print("   G = Set all lights to GREEN")
        print("   X = Emergency stop")
        print("   Q = Quit")
        print("\n💡 Detection: 3 Traffic Lights + 2 Stop Signs")
        print("="*70 + "\n")
        
        try:
            while True:
                # Get camera image
                success, image_data = qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success and image_data is not None:
                    # Convert to OpenCV format
                    img = np.frombuffer(image_data, dtype=np.uint8).reshape((410, 820, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Get position
                    result = qcar.get_world_transform()
                    
                    if result and len(result) == 4 and result[0]:
                        car_x, car_y, car_yaw = result[1][0], result[1][1], result[2][2]
                        
                        # Apply manual controls
                        self.apply_controls()
                        
                        if use_hardware:
                            qcar_hw.write(self.speed, self.steering)
                        
                        # Handle action keys
                        if self.action_keys['x']:
                            self.speed = 0.0
                            self.steering = 0.0
                            self.action_keys['x'] = False
                            print("🛑 Emergency stop!")
                        
                        if self.action_keys['r']:
                            self.set_all_lights(0)  # RED
                            self.action_keys['r'] = False
                        
                        if self.action_keys['g']:
                            self.set_all_lights(1)  # GREEN
                            self.action_keys['g'] = False
                        
                        if self.action_keys['q']:
                            break
                        
                        # Run BOTH detections
                        light_detections = self.detect_lights(img, conf_threshold=0.3)
                        sign_detections = self.detect_signs(img, conf_threshold=0.3)
                        
                        # Draw all detections
                        img_display = self.draw_detections(img.copy(), light_detections, sign_detections)
                        
                        # Overlay for info
                        overlay = img_display.copy()
                        cv2.rectangle(overlay, (0, 0), (820, 200), (0, 0, 0), -1)
                        img_display = cv2.addWeighted(overlay, 0.7, img_display, 0.3, 0)
                        
                        # Title
                        cv2.putText(img_display, "LIGHTS + STOP SIGNS", (10, 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Traffic light info
                        actual_state = "RED" if self.current_light_state == 0 else "GREEN"
                        actual_color = (0, 0, 255) if self.current_light_state == 0 else (0, 255, 0)
                        cv2.putText(img_display, f"Lights: {actual_state}", (10, 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, actual_color, 2)
                        
                        if light_detections:
                            best_light = max(light_detections, key=lambda x: x[2])
                            detected_name = self.light_class_names[best_light[1]]
                            detected_conf = best_light[2]
                            detected_color = self.light_class_colors[best_light[1]]
                            
                            cv2.putText(img_display, f"Detected: {detected_name} ({detected_conf:.2f})", 
                                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detected_color, 2)
                            
                            # Correctness
                            if detected_name == f"{actual_state}_LIGHT":
                                cv2.putText(img_display, "CORRECT", (300, 85),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else:
                                cv2.putText(img_display, "WRONG", (300, 85),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Sign info
                        if sign_detections:
                            best_sign = max(sign_detections, key=lambda x: x[2])
                            sign_name = self.sign_class_names[best_sign[1]]
                            sign_conf = best_sign[2]
                            sign_color = self.sign_class_colors[best_sign[1]]
                            
                            cv2.putText(img_display, f"Sign: {sign_name} ({sign_conf:.2f})", 
                                       (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sign_color, 2)
                        
                        # Counts
                        cv2.putText(img_display, f"Detections: {len(light_detections)} lights, {len(sign_detections)} signs", 
                                   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Position
                        cv2.putText(img_display, f"Position: ({car_x:.1f}, {car_y:.1f})", 
                                   (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Nearest light
                        nearest_light = min(self.light_positions, key=lambda x: np.hypot(car_x - x[1], car_y - x[2]))
                        dist_light = np.hypot(car_x - nearest_light[1], car_y - nearest_light[2])
                        
                        # Nearest sign
                        nearest_sign = min(self.sign_positions, key=lambda x: np.hypot(car_x - x[1], car_y - x[2]))
                        dist_sign = np.hypot(car_x - nearest_sign[1], car_y - nearest_sign[2])
                        
                        cv2.putText(img_display, f"Near: {nearest_light[0]}({dist_light:.1f}m) {nearest_sign[0]}({dist_sign:.1f}m)", 
                                   (350, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        cv2.imshow('🚦🛑 Combined Detector - Lights & Stop Signs', img_display)
                
                cv2.waitKey(1)
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if use_hardware:
                qcar_hw.write(0, 0)
            
            self.listener.stop()
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            print("\n✅ Test complete!\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚦🛑 Combined Detector - Lights & Stop Signs")
    print("="*70)
    
    # Get model paths
    if len(sys.argv) > 2:
        light_model_path = sys.argv[1]
        sign_model_path = sys.argv[2]
    else:
        light_model_path = r"traffic_light_training\traffic_light_detector\weights\best.pt"
        sign_model_path = r"traffic_sign_training\traffic_sign_detector\weights\best.pt"
        print(f"\n📂 Using default models:")
        print(f"   Lights: {light_model_path}")
        print(f"   Signs:  {sign_model_path}")
    
    print(f"\n🎯 Detection Classes:")
    print(f"   🚦 Red Light, Green Light")
    print(f"   🛑 Stop Signs")
    
    # Validate paths
    if not os.path.exists(light_model_path):
        print(f"\n❌ Light model not found: {light_model_path}")
        exit()
    
    if not os.path.exists(sign_model_path):
        print(f"\n❌ Sign model not found: {sign_model_path}")
        exit()
    
    # Create detector and run
    detector = CombinedDetector(light_model_path, sign_model_path)
    detector.run_detection_test()