import cv2
import numpy as np
import time
import sys
from io import StringIO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels
from pynput import keyboard

# Environment objects - ROAD SIGNS
from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.yield_sign import QLabsYieldSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

class WaypointRecorder:
    def __init__(self):
        print("\n🎬 WAYPOINT RECORDER - HOLD KEYS TO DRIVE!")
        print("WITH TRAFFIC SIGNS & LIGHTS!")
        print("="*70)
        
        self.qlabs = QuanserInteractiveLabs()
        self.qlabs.open("localhost")
        self.qlabs.destroy_all_spawned_actors()
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        
        # Spawn at your specified position
        print("\n🚗 Spawning QCar at your location...")
        self.qcar = QLabsQCar(self.qlabs)
        self.qcar.spawn_id(
            actorNumber=0,
            location=[-20.037, 30.423, 0.005],
            rotation=[0, 0, 300],
            waitForConfirmation=True
        )
        print("✅ QCar spawned!")
        
        # 🚦 SPAWN ROAD SIGNS AND TRAFFIC ELEMENTS
        print("\n🚦 Spawning traffic signs and lights...")
        self.spawn_traffic_elements()
        print("✅ Traffic elements spawned!")
        
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        
        # Initialize hardware control
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
            print("⚠️  No hardware control")
        
        self.waypoints = []
        
        # Driving control variables
        self.speed = 0.0
        self.steering = 0.0
        self.max_speed = 0.04
        self.max_steering = 0.3
        self.speed_increment = 0.001  # Per frame when key held
        self.steering_increment = 0.01  # Per frame when key held
        
        # 🎮 KEY STATE TRACKING - TRUE simultaneous support!
        self.keys_held = {
            'w': False, 's': False,
            'a': False, 'd': False
        }
        
        # Recording control keys
        self.recording_keys = {
            'r': False, 'space': False,
            'v': False, 'c': False,
            'x': False, 'q': False
        }
        
        # 🎯 AUTOMATIC WAYPOINT RECORDING
        self.auto_record = False
        self.min_distance = 0.3
        self.last_recorded_pos = None
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.listener.start()
        
        print("✅ Keyboard listener started")
    
    def spawn_traffic_elements(self):
        """Spawn traffic lights, signs, and crosswalks at specified positions"""
        
        # 🚶 CROSSWALKS (4 total)
        walks = []
        for i in range(4):
            walks.append(QLabsCrosswalk(self.qlabs))
        
        walks[0].spawn(location=[-5, 9.5, 0],
                        rotation=[0,0,np.pi/2], scale=[1,1,0.75],
                        configuration=0)
        walks[1].spawn(location=[1.3, 16, 0],
                    rotation=[0,0,0], scale=[1,1,0.75],
                    configuration=0)
        walks[2].spawn(location=[7.7, 9.5, 0],
                rotation=[0,0,np.pi/2], scale=[1,1,0.75],
                configuration=0)
        walks[3].spawn(location=[1.3, 3, 0],
                rotation=[0,0,0], scale=[1,1,0.75],
                configuration=0)
        
        # 🚦 TRAFFIC LIGHTS (4 total)
        lights = []
        for i in range(4):
            lights.append(QLabsTrafficLight(self.qlabs))
        
        lights[0].spawn(location=[-3.77, 13, 0],
                        rotation=[0,0,np.pi/2],
                        configuration=0)
        lights[1].spawn(location=[4.9, 14.8, 0],
                    rotation=[0,0,0],
                    configuration=0)
        lights[2].spawn(location=[6.7, 5.7, 0],
                rotation=[0,0,-np.pi/2],
                configuration=0)
        lights[3].spawn(location=[-2, 4.27, 0],
                rotation=[0,0,np.pi],
                configuration=0)
        
        # ⚠️ YIELD SIGN (1 total)
        yieldSign = QLabsYieldSign(self.qlabs)
        yieldSign.spawn(location=[0.4,-13, 0],
                            rotation=[0,0,np.pi])
        
        # 🔄 ROUNDABOUT SIGNS (3 total)
        roundAboutSigns = []
        for i in range(3):
            roundAboutSigns.append(QLabsRoundaboutSign(self.qlabs))
        
        roundAboutSigns[0].spawn(location=[24.5,33, 0],
                            rotation=[0,0,-np.pi/2])
        roundAboutSigns[1].spawn(location=[4.5,40, 0],
                            rotation=[0,0,np.pi])
        roundAboutSigns[2].spawn(location=[10.6,28.5, 0],
                            rotation=[0,0,np.pi])
        
        # 🛑 STOP SIGN (1 total)
        stop = QLabsStopSign(self.qlabs)
        stop.spawn(location=[-0.508, -7.327, 0.2], rotation=[0,0, np.pi/2],
                scale=[1,1,1], configuration=0, waitForConfirmation=True)
        
        print("   ✅ Crosswalks: 4")
        print("   ✅ Traffic Lights: 4")
        print("   ✅ Yield Sign: 1")
        print("   ✅ Roundabout Signs: 3")
        print("   ✅ Stop Sign: 1")
        
    def on_key_press(self, key):
        """Called when a key is pressed"""
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = True
                elif key.char in ['r', 'v', 'c', 'x', 'q']:
                    self.recording_keys[key.char] = True
        except AttributeError:
            if key == keyboard.Key.space:
                self.recording_keys['space'] = True
    
    def on_key_release(self, key):
        """Called when a key is released"""
        try:
            if hasattr(key, 'char'):
                if key.char in ['w', 's', 'a', 'd']:
                    self.keys_held[key.char] = False
                elif key.char in ['r', 'v', 'c', 'x']:
                    self.recording_keys[key.char] = False
        except AttributeError:
            if key == keyboard.Key.space:
                self.recording_keys['space'] = False
    
    def apply_controls(self):
        """Apply control inputs - HOLD keys to continuously accelerate!"""
        # Speed control - HOLD W to keep accelerating!
        if self.keys_held['w']:
            self.speed = min(self.speed + self.speed_increment, self.max_speed)
        elif self.keys_held['s']:
            self.speed = max(self.speed - self.speed_increment, -self.max_speed)
        else:
            # Natural decay when no input
            self.speed *= 0.97
            if abs(self.speed) < 0.001:
                self.speed = 0.0
        
        # Steering control - HOLD A/D to keep turning!
        if self.keys_held['a']:
            self.steering = min(self.steering + self.steering_increment, self.max_steering)
        elif self.keys_held['d']:
            self.steering = max(self.steering - self.steering_increment, -self.max_steering)
        else:
            # Natural decay when no input
            self.steering *= 0.9
            if abs(self.steering) < 0.01:
                self.steering = 0.0
    
    def smooth_waypoints(self, waypoints, window_size=5):
        """Apply moving average smoothing to waypoints"""
        if len(waypoints) < window_size:
            return waypoints
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(waypoints)):
            start = max(0, i - half_window)
            end = min(len(waypoints), i + half_window + 1)
            
            window_points = waypoints[start:end]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def thin_waypoints(self, waypoints, target_distance=0.5):
        """Thin waypoints to maintain consistent spacing"""
        if len(waypoints) < 2:
            return waypoints
        
        thinned = [waypoints[0]]
        
        for i in range(1, len(waypoints)):
            last_kept = thinned[-1]
            current = waypoints[i]
            
            distance = np.hypot(current[0] - last_kept[0], current[1] - last_kept[1])
            
            if distance >= target_distance:
                thinned.append(current)
        
        return thinned
    
    def record(self):
        print("\n🎮 CONTROLS - HOLD KEYS!")
        print("   ⬆️  HOLD W = Accelerate forward")
        print("   ⬇️  HOLD S = Accelerate backward")
        print("   ⬅️  HOLD A = Steer left")
        print("   ➡️  HOLD D = Steer right")
        print("   X       = Emergency stop")
        print("   💡 TIP: Hold W+A together for smooth curves!")
        print("\n📍 RECORDING:")
        print("   R     = Toggle AUTO-RECORD")
        print("   SPACE = Manual record waypoint")
        print("   V     = Save waypoints")
        print("   C     = Clear waypoints")
        print("   Q     = Quit")
        print(f"\n⚙️  AUTO-RECORD: {self.min_distance}m spacing\n")
        
        try:
            while True:
                # Get camera image
                success_img, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success_img and image_data is not None:
                    img = np.frombuffer(image_data, dtype=np.uint8)
                    img = img.reshape((410, 820, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Get car position
                    result = self.qcar.get_world_transform()
                    
                    if result and len(result) == 4:
                        success = result[0]
                        position = result[1]
                        
                        if success and position:
                            car_x = position[0]
                            car_y = position[1]
                            
                            # Apply driving controls continuously
                            self.apply_controls()
                            
                            # Handle recording key presses (one-shot)
                            if self.recording_keys['x']:
                                self.speed = 0.0
                                self.steering = 0.0
                                self.recording_keys['x'] = False
                            
                            if self.recording_keys['r']:
                                self.auto_record = not self.auto_record
                                if self.auto_record:
                                    print("\n🔴 AUTO-RECORD: ON")
                                    self.last_recorded_pos = None
                                else:
                                    print("\n⚪ AUTO-RECORD: OFF")
                                self.recording_keys['r'] = False
                            
                            if self.recording_keys['space']:
                                self.waypoints.append((car_x, car_y))
                                self.last_recorded_pos = (car_x, car_y)
                                print(f"✅ Manual #{len(self.waypoints)}: ({car_x:.3f}, {car_y:.3f})")
                                self.recording_keys['space'] = False
                            
                            if self.recording_keys['c']:
                                self.waypoints = []
                                self.last_recorded_pos = None
                                print("\n🗑️  Cleared all waypoints")
                                self.recording_keys['c'] = False
                            
                            if self.recording_keys['q']:
                                break
                            
                            if self.recording_keys['v']:
                                if len(self.waypoints) > 0:
                                    self.save_waypoints()
                                else:
                                    print("\n⚠️  No waypoints to save!")
                                self.recording_keys['v'] = False
                            
                            # Auto-record logic
                            if self.auto_record:
                                if self.last_recorded_pos is None:
                                    self.waypoints.append((car_x, car_y))
                                    self.last_recorded_pos = (car_x, car_y)
                                else:
                                    distance = np.hypot(car_x - self.last_recorded_pos[0], 
                                                       car_y - self.last_recorded_pos[1])
                                    
                                    if distance >= self.min_distance:
                                        self.waypoints.append((car_x, car_y))
                                        self.last_recorded_pos = (car_x, car_y)
                            
                            # Send control commands
                            if self.use_hardware:
                                self.qcar_hw.write(self.speed, self.steering)
                            
                            # Visualization
                            overlay = img.copy()
                            cv2.rectangle(overlay, (0, 0), (820, 180), (0, 0, 0), -1)
                            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                            
                            # Auto-record indicator
                            if self.auto_record:
                                cv2.circle(img, (780, 30), 15, (0, 0, 255), -1)
                                cv2.putText(img, "AUTO-REC", (690, 37),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Display info
                            cv2.putText(img, f"Position: ({car_x:.2f}, {car_y:.2f})", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.putText(img, f"Speed: {self.speed:.3f} | Steering: {self.steering:.2f}", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(img, f"Waypoints: {len(self.waypoints)}", (10, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Key state indicators
                            key_text = "Keys: "
                            if self.keys_held['w']: key_text += "W "
                            if self.keys_held['s']: key_text += "S "
                            if self.keys_held['a']: key_text += "A "
                            if self.keys_held['d']: key_text += "D "
                            cv2.putText(img, key_text, (10, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Distance to next waypoint
                            if self.auto_record and self.last_recorded_pos:
                                dist = np.hypot(car_x - self.last_recorded_pos[0], 
                                               car_y - self.last_recorded_pos[1])
                                bar_width = int((dist / self.min_distance) * 200)
                                bar_width = min(bar_width, 200)
                                color = (0, 255, 0) if dist < self.min_distance else (0, 165, 255)
                                cv2.rectangle(img, (10, 140), (10 + bar_width, 155), color, -1)
                            
                            cv2.putText(img, "HOLD W+A/D together! R=AutoRec SPACE=Rec V=Save X=Stop Q=Quit", (10, 175),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                            # Speed bar
                            if abs(self.speed) > 0.001:
                                bar_length = int(abs(self.speed) / self.max_speed * 200)
                                color = (0, 255, 0) if self.speed > 0 else (0, 0, 255)
                                cv2.rectangle(img, (610, 130), (610 + bar_length, 145), color, -1)
                            
                            # Steering indicator
                            steer_x = int(710 + (self.steering / self.max_steering * 100))
                            cv2.circle(img, (steer_x, 130), 8, (255, 255, 0), -1)
                            cv2.line(img, (610, 130), (810, 130), (100, 100, 100), 2)
                            
                            cv2.imshow('🚗 HOLD Keys to Drive - With Traffic Signs! 🚦', img)
                
                cv2.waitKey(1)  # Just for OpenCV window refresh
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if self.use_hardware:
                self.qcar_hw.write(0, 0)
            
            self.listener.stop()
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            print(f"\n✅ Session complete! Total waypoints: {len(self.waypoints)}\n")
    
    def save_waypoints(self):
        """Save waypoints with processing"""
        cv2.destroyAllWindows()
        
        if self.use_hardware:
            self.qcar_hw.write(0, 0)
        
        print(f"\n📊 Processing {len(self.waypoints)} waypoints...")
        
        print("\n🔧 PROCESSING OPTIONS:")
        smooth_input = input("Apply smoothing? (y/n, default=y): ").strip().lower()
        do_smooth = smooth_input != 'n'
        
        thin_input = input("Thin waypoints? (y/n, default=y): ").strip().lower()
        do_thin = thin_input != 'n'
        
        if do_thin:
            target_dist = input(f"Target distance (default={self.min_distance}m): ").strip()
            target_dist = float(target_dist) if target_dist else self.min_distance
        
        processed = self.waypoints.copy()
        
        if do_smooth:
            window = 5
            processed = self.smooth_waypoints(processed, window)
            print(f"✅ Applied smoothing (window={window})")
        
        if do_thin:
            before_count = len(processed)
            processed = self.thin_waypoints(processed, target_dist)
            print(f"✅ Thinned: {before_count} → {len(processed)} waypoints")
        
        print("\n")
        filename = input("💾 Enter filename (e.g., 'roundabout'): ").strip()
        if filename:
            np.save(f'{filename}_waypoints.npy', np.array(processed))
            print(f"✅ Saved {len(processed)} PROCESSED waypoints to {filename}_waypoints.npy")
            
            np.save(f'{filename}_waypoints_raw.npy', np.array(self.waypoints))
            print(f"✅ Saved {len(self.waypoints)} RAW waypoints to {filename}_waypoints_raw.npy")
            
            with open(f'{filename}_waypoints.txt', 'w') as f:
                f.write(f"# {filename} waypoints\n")
                f.write(f"# RAW: {len(self.waypoints)} points\n")
                f.write(f"# PROCESSED: {len(processed)} points\n")
                f.write(f"# Start: ({processed[0][0]:.3f}, {processed[0][1]:.3f})\n")
                f.write(f"# End: ({processed[-1][0]:.3f}, {processed[-1][1]:.3f})\n\n")
                for i, (x, y) in enumerate(processed):
                    f.write(f"{i:3d}: ({x:8.3f}, {y:8.3f})\n")
            
            print(f"✅ Text file: {filename}_waypoints.txt")
            print("\n✨ Ready to record next zone!")
            print("📹 Resuming...\n")

if __name__ == '__main__':
    try:
        recorder = WaypointRecorder()
        recorder.record()
    except ImportError:
        print("\n❌ ERROR: pynput library not found!")
        print("\n📦 Please install it:")
        print("   pip install pynput")
        print("\nOr if using conda:")
        print("   conda install -c conda-forge pynput")
        print("\n")