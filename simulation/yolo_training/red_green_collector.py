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

# Environment objects - CLEAN VERSION (matching lane follower)
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

class StopSignCollector:
    def __init__(self):
        print("\n" + "="*70)
        print("🛑 STOP SIGN COLLECTOR")
        print("📸 Manual driving + image capture")
        print("="*70)
        
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
        
        # MANUAL DRIVING CONTROLS
        self.speed = 0.0
        self.steering = 0.0
        self.max_speed = 0.04
        self.max_steering = 0.3
        self.speed_increment = 0.001
        self.steering_increment = 0.01
        
        # Key states
        self.keys_held = {'w': False, 's': False, 'a': False, 'd': False}
        self.action_keys = {'c': False, 'x': False, 'q': False}
        
        # Dataset folder
        self.save_dir = "traffic_signs_dataset"
        self.stop_dir = os.path.join(self.save_dir, "stop_signs")
        self.annotations_dir = os.path.join(self.save_dir, "annotations")
        
        os.makedirs(self.stop_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        self.frame_metadata = []
        self.collection_start_time = datetime.now()
        
        # COUNT EXISTING IMAGES TO CONTINUE NUMBERING (DON'T OVERWRITE!)
        existing_stop = len([f for f in os.listdir(self.stop_dir) 
                            if f.startswith('stop_') and f.endswith('.png')])
        
        self.stop_count = existing_stop
        
        if existing_stop > 0:
            print(f"\n📁 Found existing images!")
            print(f"   Stop signs: {existing_stop}")
            print(f"   Continuing from: Stop #{existing_stop+1}")
        
        # Keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        
        print(f"\n📁 Dataset folder:")
        print(f"   Stop signs: {os.path.abspath(self.stop_dir)}")
    
    def spawn_traffic_elements(self):
        """Spawn traffic elements - MATCHING LANE FOLLOWER EXACTLY"""
        
        # Crosswalks - 4 total
        self.walks = [QLabsCrosswalk(self.qlabs) for _ in range(4)]
        self.walks[0].spawn(location=[-5, 9.5, 0], rotation=[0,0,np.pi/2], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[1].spawn(location=[1.3, 16, 0], rotation=[0,0,0], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[2].spawn(location=[7.7, 9.5, 0], rotation=[0,0,np.pi/2], 
                           scale=[1,1,0.75], configuration=0)
        self.walks[3].spawn(location=[1.3, 3, 0], rotation=[0,0,0], 
                           scale=[1,1,0.75], configuration=0)
        
        # Traffic lights - 3 total (lowered -2.5m, scaled 1.8x)
        self.lights = [QLabsTrafficLight(self.qlabs) for _ in range(3)]
        self.lights[0].spawn(location=[-23.513, 26.363, -2.5], rotation=[0,0,135], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[1].spawn(location=[-2.152, 3.8, -2.5], rotation=[0,0,300], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        self.lights[2].spawn(location=[26.687, 8.74, -2.5], rotation=[0,0,0], 
                            scale=[1.8, 1.8, 1.8], configuration=0)
        
        # STOP SIGNS - 2 total (EXACT positions from lane follower)
        QLabsStopSign(self.qlabs).spawn(location=[-0.508, -7.327, 0.2], 
                                        rotation=[0,0, np.pi/2], scale=[1,1,1], 
                                        configuration=0, waitForConfirmation=True)
        QLabsStopSign(self.qlabs).spawn(location=[24.5, 33, 0.2], 
                                        rotation=[0,0,-np.pi/2], scale=[1,1,1], 
                                        configuration=0, waitForConfirmation=True)
        
        print("   ✅ Crosswalks: 4")
        print("   ✅ Traffic Lights: 3 (lowered -2.5m, scaled 1.8x)")
        print("   ✅ Stop Signs: 2")
        print("\n🛑 STOP SIGN LOCATIONS (2 signs total):")
        print("   STOP1: (-0.508, -7.327)")
        print("   STOP2: (24.5, 33.0)")
    
    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = True
                elif key.char in ['c', 'x', 'q']:
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
        """Apply manual control inputs - HOLD keys to drive!"""
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
        """Save image to stop signs folder"""
        self.stop_count += 1
        filename = f"stop_{self.stop_count:04d}.png"
        filepath = os.path.join(self.stop_dir, filename)
        
        cv2.imwrite(filepath, img)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'sign_type': 'stop',
            'timestamp': datetime.now().isoformat(),
            'car_position': {'x': float(car_x), 'y': float(car_y), 'yaw': float(car_yaw)},
            'folder': 'stop_signs'
        }
        self.frame_metadata.append(metadata)
        
        print(f"📸 Captured: Stop #{self.stop_count} at ({car_x:.1f}, {car_y:.1f})")
    
    def finalize(self):
        """Save metadata and summary"""
        metadata_file = os.path.join(self.annotations_dir, 'stop_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.frame_metadata, f, indent=2)
        
        summary = {
            'total_images': self.stop_count,
            'stop_signs': self.stop_count,
            'collection_start': self.collection_start_time.isoformat(),
            'collection_end': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.collection_start_time).total_seconds()
        }
        
        summary_file = os.path.join(self.save_dir, 'stop_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print("✅ COLLECTION COMPLETE!")
        print(f"{'='*70}")
        print(f"   🛑 Stop signs: {summary['stop_signs']} images")
        print(f"   📁 Location: {os.path.abspath(self.save_dir)}")
        print(f"{'='*70}\n")
    
    def run(self):
        print("\n" + "="*70)
        print("🎮 MANUAL CONTROLS:")
        print("="*70)
        print("DRIVING:")
        print("   ⬆️  HOLD W = Forward")
        print("   ⬇️  HOLD S = Backward")
        print("   ⬅️  HOLD A = Steer Left")
        print("   ➡️  HOLD D = Steer Right")
        print("   💡 TIP: Hold W+A for smooth left turns!")
        print("\nCAPTURE:")
        print("   📸 C = Capture current image")
        print("\nOTHER:")
        print("   🛑 X = Emergency stop")
        print("   ❌ Q = Quit and save")
        print("\n📁 Images saved to:")
        print(f"   {self.stop_dir}/")
        print("\n🎯 TARGET: Collect 150-200 stop sign images!")
        print("   Visit both stop signs (see locations above)")
        print("   Capture from different angles and distances")
        print("="*70 + "\n")
        
        try:
            while True:
                # Get image
                success_img, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success_img and image_data is not None:
                    img = np.frombuffer(image_data, dtype=np.uint8).reshape((410, 820, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Get position
                    result = self.qcar.get_world_transform()
                    
                    if result and len(result) == 4 and result[0]:
                        car_x, car_y, car_yaw = result[1][0], result[1][1], result[2][2]
                        
                        # Apply manual driving controls
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
                        
                        if self.action_keys['q']:
                            break
                        
                        # Visualization
                        overlay = img.copy()
                        cv2.rectangle(overlay, (0, 0), (820, 140), (0, 0, 0), -1)
                        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                        
                        # Stop sign indicator (red octagon)
                        cv2.circle(img, (750, 50), 30, (0, 0, 255), -1)  # Red
                        cv2.circle(img, (750, 50), 33, (255, 255, 255), 3)
                        cv2.putText(img, "STOP", (720, 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, "🛑 STOP SIGNS", (600, 95),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Info
                        cv2.putText(img, f"Position: ({car_x:.2f}, {car_y:.2f})", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(img, f"Stop Sign Images: {self.stop_count}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(img, f"Speed: {self.speed:.3f} | Steering: {self.steering:.2f}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Key states
                        key_text = "Keys: "
                        if self.keys_held['w']: key_text += "W "
                        if self.keys_held['s']: key_text += "S "
                        if self.keys_held['a']: key_text += "A "
                        if self.keys_held['d']: key_text += "D "
                        cv2.putText(img, key_text, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        cv2.imshow('🛑 Stop Sign Collector', img)
                
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
            
            if self.stop_count > 0:
                self.finalize()
            else:
                print("\n⚠️  No images captured")
            
            print("\n✅ Session complete!\n")

if __name__ == '__main__':
    try:
        collector = StopSignCollector()
        collector.run()
    except ImportError:
        print("\n❌ ERROR: pynput not found!")
        print("   pip install pynput\n")