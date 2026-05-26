import cv2
import numpy as np
import time
import sys
import os
import json
from io import StringIO
from datetime import datetime
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

# Environment objects - ROAD SIGNS ONLY
from qvl.crosswalk import QLabsCrosswalk
from qvl.roundabout_sign import QLabsRoundaboutSign
from qvl.yield_sign import QLabsYieldSign
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

# ==================== DATA COLLECTION CLASS ====================
class TrafficDatasetCollector:
    def __init__(self, save_dir="traffic_dataset"):
        self.save_dir = save_dir
        self.images_dir = os.path.join(save_dir, "images")
        self.annotations_dir = os.path.join(save_dir, "annotations")
        
        # Create directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        self.frame_metadata = []
        self.collection_start_time = datetime.now()
        
        print(f"\n📁 Dataset Directory: {os.path.abspath(save_dir)}")
        print(f"   📷 Images: {self.images_dir}")
        print(f"   📋 Annotations: {self.annotations_dir}")
    
    def save_frame(self, img, frame_count, car_x, car_y, car_yaw, light_states):
        """Save frame with metadata for training - ONLY traffic lights, stop sign, and roundabout signs"""
        # Save image as PNG (lossless, better quality than JPG)
        filename = f"frame_{frame_count:06d}.png"
        filepath = os.path.join(self.images_dir, filename)
        cv2.imwrite(filepath, img)  # PNG is lossless by default
        
        # Save metadata - ONLY relevant signs
        metadata = {
            'frame': frame_count,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'car_position': {
                'x': float(car_x),
                'y': float(car_y),
                'yaw': float(car_yaw)
            },
            'traffic_lights': {
                'light_0': {'state': self._get_light_state_name(light_states[0]), 'location': [-22.313, 36.363, 0.0]},
                'light_1': {'state': self._get_light_state_name(light_states[1]), 'location': [-2.95, 5.6, 0]},
                'light_2': {'state': self._get_light_state_name(light_states[2]), 'location': [6.7, 5.7, 0]},
                'light_3': {'state': self._get_light_state_name(light_states[3]), 'location': [24.387, 4.74, 0.2]}
            },
            'signs': {
                'stop_sign': {'location': [-0.508, -7.327, 0.2]},
                'roundabout_signs': [
                    {'location': [24.5, 33, 0]},
                    {'location': [4.5, 40, 0]},
                    {'location': [10.6, 28.5, 0]}
                ]
            }
        }
        
        self.frame_metadata.append(metadata)
        
        # Save metadata every 50 frames
        if frame_count % 50 == 0:
            self._save_metadata()
            print(f"💾 Saved {frame_count} frames to dataset ({len(self.frame_metadata)} total)")
    
    def _get_light_state_name(self, state):
        """Convert state number to name"""
        state_names = {0: 'RED', 1: 'YELLOW', 2: 'GREEN'}
        return state_names.get(state, 'UNKNOWN')
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_file = os.path.join(self.annotations_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.frame_metadata, f, indent=2)
    
    def finalize(self):
        """Finalize dataset collection"""
        self._save_metadata()
        
        # Save summary - ONLY relevant classes
        summary = {
            'total_frames': len(self.frame_metadata),
            'collection_start': self.collection_start_time.isoformat(),
            'collection_end': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.collection_start_time).total_seconds(),
            'classes': [
                'traffic_light_red',
                'traffic_light_yellow', 
                'traffic_light_green',
                'stop_sign',
                'roundabout_sign'
            ],
            'sign_locations': {
                'traffic_lights': 4,
                'stop_signs': 1,
                'roundabout_signs': 3
            }
        }
        
        summary_file = os.path.join(self.save_dir, 'dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "="*70)
        print("✅ DATASET COLLECTION COMPLETE!")
        print("="*70)
        print(f"   📊 Total frames: {summary['total_frames']}")
        print(f"   ⏱️  Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"   📁 Location: {os.path.abspath(self.save_dir)}")
        print(f"\n📋 CLASSES TO LABEL (6 total):")
        for i, cls in enumerate(summary['classes'], 1):
            print(f"   {i}. {cls}")
        print(f"\n📋 NEXT STEP: Label Your Images")
        print(f"   Option 1 (Recommended): Use Roboflow (AI-assisted)")
        print(f"      https://roboflow.com")
        print(f"      - Upload images")
        print(f"      - Use AI-assisted labeling")
        print(f"      - Export in YOLO format")
        print(f"   Option 2: Install LabelImg")
        print(f"      pip install labelImg")
        print(f"      labelImg {self.images_dir}")
        print(f"\n   ⏱️  Labeling time estimate:")
        print(f"      - With Roboflow AI: ~30-60 minutes")
        print(f"      - Manual labeling: ~2-4 hours")
        print("="*70 + "\n")


# ==================== LANE FOLLOWER CLASS ====================
class LaneCenterFollower:
    def __init__(self):
        print("\n" + "="*70)
        print("LANE CENTERING - DUAL WAYPOINTS (ROUND16 + CURVE12) - SMOOTH MODE!")
        print("WITH TRAFFIC DATASET COLLECTION 📊")
        print("Classes: Traffic Lights (R/Y/G), Stop Sign, Roundabout Sign")
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
        
        # 🚦 SPAWN ROAD SIGNS AND TRAFFIC ELEMENTS
        print("\n🚦 Spawning traffic signs and lights...")
        self.spawn_traffic_elements()
        print("✅ Traffic elements spawned!")
        
        print("\n▶️  Starting simulation...")
        QLabsRealTime().start_real_time_model(rtmodels.QCAR)
        time.sleep(2)
        print("✅ Running!")
        
        # Speed control - with 5 second transition
        self.initial_speed = 0.035  # Start slow
        self.cruising_speed = 0.045  # Full speed after 5 seconds
        self.speed_transition_time = 5.0  # 5 second delay
        self.start_time = None
        
        # 🐌 TIMED SLOW ZONE at specific position
        self.slow_zone_center = (-20.02, 23.332)  # Target position
        self.slow_zone_radius = 2.0  # Trigger radius (meters)
        self.slow_zone_speed = 0.035  # Slow speed in zone (SMOOTH: 0.045 → 0.035)
        self.slow_zone_duration = 3.0  # Duration in seconds (3 seconds)
        self.slow_zone_start_time = None  # When we entered
        self.in_slow_zone = False  # Are we in slow zone?
        self.slow_zone_completed = False  # Have we finished slow zone?
        
        # 🗺️ DUAL WAYPOINT TRACKING - Round16 AND Curve12
        self.round16_waypoints = self.load_waypoints('round16_waypoints.npy')
        self.curve12_waypoints = self.load_waypoints('curve12_waypoints.npy')
        
        # Active waypoint set (switches between round16 and curve12)
        self.active_waypoints = None
        self.active_waypoint_name = None
        
        # CURVE12 completion tracking
        self.curve12_completed = False
        self.curve12_completion_time = None
        self.post_curve12_slowdown_duration = 5.0  # 5 seconds after curve12 completes
        
        self.in_special_zone = False
        self.current_waypoint_idx = 0
        self.lookahead = 0.5  # meters - REDUCED from 1.0 to prevent corner cutting
        self.waypoint_L = 0.256  # QCar wheelbase
        self.special_zone_speed = 0.045  # Faster speed for waypoint mode (ROUND16)
        # No separate CURVE12 speed - uses normal cruising speed (0.045)
        
        # 🚦 CYAN LINE DETECTION - Exit waypoint mode when cyan STARTS (appears)
        self.cyan_start_threshold = 500  # If cyan pixels > this, consider cyan started
        self.cyan_start_counter = 0  # Count frames with cyan
        self.cyan_start_frames = 5  # Need 5 consecutive frames to confirm cyan started
        
        # Control parameters
        self.kp = 0.8
        self.kd = 0.2
        self.previous_error = 0
        
        # ROI SETTINGS
        self.roi_start = 0.55
        self.roi_height = 0.45
        
        # 🔧 FIXED HSV VALUES
        # CYAN - Wider range for better detection
        self.lower_cyan = np.array([80, 0, 150])
        self.upper_cyan = np.array([170, 255, 255])
        
        # WHITE - MUCH MORE RESTRICTIVE (only bright white!)
        self.lower_white = np.array([0, 0, 240])
        self.upper_white = np.array([180, 25, 255])
        
        # 🎯 FALLBACK STRATEGY - Distance from edges when one is missing
        self.offset_from_white = 240  # Stay 240px LEFT of white line
        self.offset_from_cyan = 250   # Stay 250px RIGHT of cyan line
        
        # 🎯 TIME-BASED CYAN FALLBACK - Switch to white-only if cyan flickers
        self.last_cyan_loss_time = None  # Track when cyan was last lost
        self.cyan_switch_timeout = 3.0   # 3 seconds timeout
        self.force_white_only = False    # Force white-only mode flag
        self.was_using_cyan = False      # Track if we were using cyan
        
        # 🎯 SMOOTH TRANSITION SYSTEM - Prevent jerky steering
        self.previous_lane_center = None  # Last detected lane center
        self.previous_strategy = None  # Last detection strategy
        self.transition_frames = 15  # Number of frames to blend (15 frames = ~0.75s at 20fps)
        self.transition_counter = 0  # Current transition frame count
        self.transition_start_center = None  # Lane center when transition started
        self.transition_target_center = None  # Target lane center after transition
        
        # 🎯 EMA FILTER for lane center smoothing
        self.ema_alpha = 0.3  # EMA smoothing factor (0.3 = 30% new, 70% old) - lower = smoother
        self.ema_lane_center = None  # Smoothed lane center
        
        # 🎯 STEERING RATE LIMITER - Prevent sudden steering changes
        self.max_steering_change = 0.05  # Maximum steering change per frame (radians)
        self.previous_steering = 0.0  # Last steering command
        
        # 🚦 TRAFFIC LIGHT CYCLING SETUP - 3 SECONDS FOR DATASET VARIETY
        self.light_cycle_interval = 3.0  # Change lights every 3 seconds (faster for more variety)
        self.last_light_cycle_time = time.time()
        
        # 📊 DATA COLLECTION SETUP
        self.collect_dataset = True  # SET TO False TO DISABLE DATA COLLECTION
        self.collection_interval = 3  # Save every 3 frames (MORE FREQUENT for better coverage near signs)
        
        if self.collect_dataset:
            self.dataset_collector = TrafficDatasetCollector(save_dir="traffic_dataset")
            print(f"\n📊 DATA COLLECTION ENABLED")
            print(f"   💾 Save interval: Every {self.collection_interval} frames")
            print(f"   🎯 Classes: Traffic Lights (R/Y/G), Stop Sign, Roundabout Sign")
            print(f"   🎯 Target: 3,000+ images (run for 10-15 laps)")
            print(f"   ⏱️  Estimated time: 30-45 minutes")
        
        print(f"\n✅ Speed: {self.initial_speed} → {self.cruising_speed} (after {self.speed_transition_time}s)")
        print(f"🐌 Slow Zone: Center=({self.slow_zone_center[0]:.2f}, {self.slow_zone_center[1]:.2f}), "
              f"Radius={self.slow_zone_radius}m, Speed={self.slow_zone_speed} (smooth transition), Duration={self.slow_zone_duration}s")
        print(f"   📊 Speed Profile: 0.035 (5s) → 0.045 → 0.035 (3s in slow zone) → 0.045")
        print(f"✅ Waypoint Speed: {self.special_zone_speed} (ROUND16 & CURVE12 - same speed)")
        print(f"✅ Lookahead Distance: {self.lookahead}m (REDUCED to prevent corner cutting)")
        print(f"✅ Cyan Start Detection: > {self.cyan_start_threshold}px for {self.cyan_start_frames} frames → Exit waypoint mode")
        
        print(f"\n🎯 SMOOTH STEERING SYSTEM:")
        print(f"   ✅ Transition Blending: {self.transition_frames} frames (~{self.transition_frames*0.05:.2f}s)")
        print(f"   ✅ EMA Filter: Alpha={self.ema_alpha} (lower = smoother)")
        print(f"   ✅ Rate Limiter: Max change={self.max_steering_change:.3f} rad/frame")
        print(f"   💡 This prevents jerky movements when switching detection modes!")
        
        print(f"\n🗺️  DUAL WAYPOINT SYSTEM:")
        if self.round16_waypoints is not None:
            print(f"   ✅ ROUND16: {len(self.round16_waypoints)} points")
            print(f"      Start: ({self.round16_waypoints[0][0]:.2f}, {self.round16_waypoints[0][1]:.2f})")
            print(f"      End: ({self.round16_waypoints[-1][0]:.2f}, {self.round16_waypoints[-1][1]:.2f})")
            print(f"      Trigger: Box (12.5, 20.0, 23.0, 45.5)")
        else:
            print(f"   ⚠️  ROUND16: Not loaded")
        
        if self.curve12_waypoints is not None:
            print(f"   ✅ CURVE12: {len(self.curve12_waypoints)} points")
            print(f"      Start: ({self.curve12_waypoints[0][0]:.2f}, {self.curve12_waypoints[0][1]:.2f})")
            print(f"      End: ({self.curve12_waypoints[-1][0]:.2f}, {self.curve12_waypoints[-1][1]:.2f})")
            print(f"      🎯 Entry: Within 5.0m of start point")
            print(f"      🔒 STRICT MODE: No vision switch, no slowdown")
            print(f"      🔄 COMPLETION: At waypoint 104 → 5s slowdown at initial speed (0.035)")
        else:
            print(f"   ⚠️  CURVE12: Not loaded")
        
        if self.round16_waypoints is None and self.curve12_waypoints is None:
            print(f"   ⚠️  No waypoints loaded - VISION ONLY mode")
        
        print(f"\n🚦 TRAFFIC LIGHT CYCLING:")
        print(f"   ✅ Interval: Every {self.light_cycle_interval} seconds (FAST for dataset variety)")
        print(f"   ✅ Cycle: RED → YELLOW → GREEN → RED")
        print(f"   ✅ Full cycle duration: ~{self.light_cycle_interval * 3:.0f} seconds")
        
        print(f"\n✅ ROI: {self.roi_start} to {self.roi_start + self.roi_height}")
        print(f"✅ CYAN HSV: [80,0,150] to [170,255,255] (WIDER RANGE)")
        print(f"✅ WHITE HSV: [0,0,240] to [180,25,255] (MORE RESTRICTIVE)")
        print(f"✅ Fallback: {self.offset_from_white}px from white, {self.offset_from_cyan}px from cyan")
        print(f"✅ Cyan Timeout: {self.cyan_switch_timeout}s - switches to WHITE-ONLY if cyan flickers")
        print("\n💡 Press 'q' to stop and save dataset\n")
    
    def spawn_traffic_elements(self):
        """Spawn traffic lights, signs, and crosswalks at specified positions"""
        
        # 🚶 CROSSWALKS (4 total) - spawned but not in dataset
        self.walks = []
        for i in range(4):
            self.walks.append(QLabsCrosswalk(self.qlabs))
        
        self.walks[0].spawn(location=[-5, 9.5, 0],
                        rotation=[0,0,np.pi/2], scale=[1,1,0.75],
                        configuration=0)
        self.walks[1].spawn(location=[1.3, 16, 0],
                    rotation=[0,0,0], scale=[1,1,0.75],
                    configuration=0)
        self.walks[2].spawn(location=[7.7, 9.5, 0],
                rotation=[0,0,np.pi/2], scale=[1,1,0.75],
                configuration=0)
        self.walks[3].spawn(location=[1.3, 3, 0],
                rotation=[0,0,0], scale=[1,1,0.75],
                configuration=0)
        
        # 🚦 TRAFFIC LIGHTS (4 total) - IN DATASET
        self.lights = []
        self.light_states = [0, 2, 1, 0]  # Initial states: RED, GREEN, YELLOW, RED
        for i in range(4):
            self.lights.append(QLabsTrafficLight(self.qlabs))
        
        self.lights[0].spawn(location=[-22.313, 36.363, 0.0],
                        rotation=[0,0,135],
                        configuration=self.light_states[0])
        self.lights[1].spawn(location=[-2.95, 5.6, 0],
                    rotation=[0,0,300],
                    configuration=self.light_states[1])
        self.lights[2].spawn(location=[6.7, 5.7, 0],
                rotation=[0,0,-np.pi/2],
                configuration=self.light_states[2])
        self.lights[3].spawn(location=[24.387, 4.74, 0.2],
                rotation=[0,0,0],
                configuration=self.light_states[3])
        
        # ⚠️ YIELD SIGN (1 total) - spawned but not in dataset
        yieldSign = QLabsYieldSign(self.qlabs)
        yieldSign.spawn(location=[0.4,-13, 0],
                            rotation=[0,0,np.pi])
        
        # 🔄 ROUNDABOUT SIGNS (3 total) - IN DATASET
        roundAboutSigns = []
        for i in range(3):
            roundAboutSigns.append(QLabsRoundaboutSign(self.qlabs))
        
        roundAboutSigns[0].spawn(location=[24.5,33, 0],
                            rotation=[0,0,-np.pi/2])
        roundAboutSigns[1].spawn(location=[4.5,40, 0],
                            rotation=[0,0,np.pi])
        roundAboutSigns[2].spawn(location=[10.6,28.5, 0],
                            rotation=[0,0,np.pi])
        
        # 🛑 STOP SIGN (1 total) - IN DATASET
        stop = QLabsStopSign(self.qlabs)
        stop.spawn(location=[-0.508, -7.327, 0.2], rotation=[0,0, np.pi/2],
                scale=[1,1,1], configuration=0, waitForConfirmation=True)
        
        state_names = ['RED', 'YELLOW', 'GREEN']
        states_display = [state_names[s] for s in self.light_states]
        print("   ✅ Traffic Lights: 4 (Initial: {}) - IN DATASET".format(states_display))
        print("   ✅ Stop Sign: 1 - IN DATASET")
        print("   ✅ Roundabout Signs: 3 - IN DATASET")
        print("   ℹ️  Crosswalks: 4 (environment only)")
        print("   ℹ️  Yield Sign: 1 (environment only)")
    
    def cycle_traffic_lights(self):
        """Cycle traffic lights through states: RED → YELLOW → GREEN → RED"""
        current_time = time.time()
        
        if current_time - self.last_light_cycle_time >= self.light_cycle_interval:
            # Cycle each light through states (0→1→2→0)
            for i, light in enumerate(self.lights):
                self.light_states[i] = (self.light_states[i] + 1) % 3
                light.set_state(self.light_states[i])
            
            # Display current states
            state_names = ['RED', 'YELLOW', 'GREEN']
            states_display = [state_names[s] for s in self.light_states]
            print(f"\n🚦 TRAFFIC LIGHTS CYCLED: {states_display}")
            
            self.last_light_cycle_time = current_time
    
    def load_waypoints(self, filename):
        """Load waypoints from file"""
        if os.path.exists(filename):
            try:
                waypoints = np.load(filename)
                print(f"✅ Loaded: {filename} ({len(waypoints)} points)")
                return waypoints
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                return None
        else:
            print(f"ℹ️  Not found: {filename}")
            return None
    
    def smooth_lane_center_transition(self, new_lane_center, new_strategy):
        """
        Smooth transition between different detection strategies
        Uses blending over multiple frames to prevent jerky movements
        """
        if new_lane_center is None:
            return None
        
        # Check if strategy changed
        strategy_changed = (self.previous_strategy is not None and 
                          self.previous_strategy != new_strategy)
        
        if strategy_changed:
            # Strategy changed! Start transition
            if self.transition_counter == 0:
                # First frame of transition
                self.transition_start_center = self.previous_lane_center
                self.transition_target_center = new_lane_center
                self.transition_counter = 1
                print(f"\n🔄 SMOOTH TRANSITION: {self.previous_strategy} → {new_strategy} ({self.transition_frames} frames)")
            
            # Blend between old and new lane centers
            if self.transition_counter < self.transition_frames:
                # Calculate blend ratio (0.0 to 1.0)
                blend_ratio = self.transition_counter / self.transition_frames
                
                # Linear interpolation between start and target
                if self.transition_start_center and self.transition_target_center:
                    blended_x = int(self.transition_start_center[0] * (1 - blend_ratio) + 
                                  self.transition_target_center[0] * blend_ratio)
                    blended_y = int(self.transition_start_center[1] * (1 - blend_ratio) + 
                                  self.transition_target_center[1] * blend_ratio)
                    blended_center = (blended_x, blended_y)
                    
                    self.transition_counter += 1
                    return blended_center
                else:
                    # Fallback if no valid start center
                    self.transition_counter = self.transition_frames
                    return new_lane_center
            else:
                # Transition complete
                self.transition_counter = 0
                self.transition_start_center = None
                self.transition_target_center = None
                return new_lane_center
        else:
            # No strategy change, reset transition
            self.transition_counter = 0
            self.transition_start_center = None
            self.transition_target_center = None
            return new_lane_center
    
    def apply_ema_filter(self, new_lane_center):
        """
        Apply Exponential Moving Average filter to lane center
        Smooths out noise and sudden jumps
        """
        if new_lane_center is None:
            return self.ema_lane_center
        
        if self.ema_lane_center is None:
            # First frame, initialize EMA
            self.ema_lane_center = new_lane_center
            return self.ema_lane_center
        
        # Apply EMA: smoothed = alpha * new + (1 - alpha) * old
        smoothed_x = int(self.ema_alpha * new_lane_center[0] + 
                        (1 - self.ema_alpha) * self.ema_lane_center[0])
        smoothed_y = int(self.ema_alpha * new_lane_center[1] + 
                        (1 - self.ema_alpha) * self.ema_lane_center[1])
        
        self.ema_lane_center = (smoothed_x, smoothed_y)
        return self.ema_lane_center
    
    def apply_steering_rate_limit(self, new_steering):
        """
        Limit the rate of change of steering to prevent sudden jerks
        """
        steering_change = new_steering - self.previous_steering
        
        # Clip the change to maximum allowed
        if abs(steering_change) > self.max_steering_change:
            limited_steering = self.previous_steering + np.sign(steering_change) * self.max_steering_change
            self.previous_steering = limited_steering
            return limited_steering
        else:
            self.previous_steering = new_steering
            return new_steering
    
    def check_timed_slow_zone(self, car_x, car_y):
        """Check if car is in timed slow zone and manage timing"""
        # Calculate distance from slow zone center
        distance = np.hypot(car_x - self.slow_zone_center[0], car_y - self.slow_zone_center[1])
        
        # If we already completed this zone, don't activate again
        if self.slow_zone_completed:
            return False, 0.0
        
        # Check if we're within the radius
        if distance <= self.slow_zone_radius:
            if not self.in_slow_zone:
                # Just entered slow zone
                self.in_slow_zone = True
                self.slow_zone_start_time = time.time()
                print(f"\n🐌 ENTERED SLOW ZONE at ({car_x:.1f}, {car_y:.1f}) - Distance={distance:.2f}m")
                print(f"   ⏱️  Starting {self.slow_zone_duration}s timer...")
            
            # Calculate elapsed time in zone
            elapsed = time.time() - self.slow_zone_start_time
            remaining = max(0, self.slow_zone_duration - elapsed)
            
            # Check if duration completed
            if elapsed >= self.slow_zone_duration:
                self.in_slow_zone = False
                self.slow_zone_completed = True
                print(f"\n✅ SLOW ZONE COMPLETED - Resuming normal speed")
                return False, 0.0
            
            return True, remaining
        
        else:
            # Outside the radius
            if self.in_slow_zone:
                # We left the zone before timer finished
                print(f"\n⚠️  LEFT SLOW ZONE EARLY at ({car_x:.1f}, {car_y:.1f})")
                self.in_slow_zone = False
                self.slow_zone_completed = True
            
            return False, 0.0
    
    def check_special_zone(self, car_x, car_y):
        """Check which waypoint zone car is in - Round16 or Curve12"""
        
        # 🎯 CURVE12 ENTRY CONDITION - Must be near starting point!
        if self.curve12_waypoints is not None and not self.in_special_zone and not self.curve12_completed:
            curve12_start = self.curve12_waypoints[0]
            distance_to_curve12_start = np.hypot(car_x - curve12_start[0], car_y - curve12_start[1])
            curve12_entry_threshold = 5.0  # Within 5 meters of curve12 start (INCREASED for easier entry)
            
            if distance_to_curve12_start <= curve12_entry_threshold:
                print(f"\n🗺️  ENTERED CURVE12 ZONE at ({car_x:.1f}, {car_y:.1f})")
                print(f"   📍 Distance to curve12 start: {distance_to_curve12_start:.2f}m")
                self.in_special_zone = True
                self.active_waypoints = self.curve12_waypoints
                self.active_waypoint_name = "CURVE12"
                self.current_waypoint_idx = 0
                self.cyan_start_counter = 0
                # Reset smoothing when entering waypoint mode
                self.ema_lane_center = None
                self.previous_steering = 0.0
                return True
        
        # 🎯 ROUND16 TRIGGER BOX - Original behavior
        if self.round16_waypoints is not None:
            round16_box = (12.5, 20.0, 23.0, 45.5)  # (min_x, min_y, max_x, max_y)
            min_x, min_y, max_x, max_y = round16_box
            
            if min_x <= car_x <= max_x and min_y <= car_y <= max_y:
                if not self.in_special_zone:
                    print(f"\n🗺️  ENTERED ROUND16 ZONE at ({car_x:.1f}, {car_y:.1f})")
                    self.in_special_zone = True
                    self.active_waypoints = self.round16_waypoints
                    self.active_waypoint_name = "ROUND16"
                    self.current_waypoint_idx = 0
                    self.cyan_start_counter = 0
                    # Reset smoothing when entering waypoint mode
                    self.ema_lane_center = None
                    self.previous_steering = 0.0
                return True
        
        # Check if still in active zone
        if self.in_special_zone:
            # For CURVE12 - check if completed (reached waypoint 104!)
            if self.active_waypoint_name == "CURVE12":
                # Check if reached waypoint 104
                if self.current_waypoint_idx >= 104:
                    print(f"\n🔄 CURVE12 COMPLETED! Reached waypoint 104 at ({car_x:.1f}, {car_y:.1f})")
                    print(f"   📍 Waypoint index: {self.current_waypoint_idx}")
                    print(f"   ⏱️  Starting 5-second slowdown at initial speed (0.035)...")
                    self.in_special_zone = False
                    self.active_waypoints = None
                    self.active_waypoint_name = None
                    self.curve12_completed = True
                    self.curve12_completion_time = time.time()
                    return False
                
                # STRICT MODE: Stay in CURVE12 until waypoint 104, no trigger box check!
                return True
            
            # For ROUND16 - use its trigger box
            elif self.active_waypoint_name == "ROUND16":
                round16_box = (12.5, 20.0, 23.0, 45.5)
                min_x, min_y, max_x, max_y = round16_box
                if min_x <= car_x <= max_x and min_y <= car_y <= max_y:
                    return True
            
            # Exited zone
            print(f"\n👁️  EXITED {self.active_waypoint_name} ZONE at ({car_x:.1f}, {car_y:.1f}), switching to VISION")
            self.in_special_zone = False
            self.active_waypoints = None
            self.active_waypoint_name = None
            self.current_waypoint_idx = 0
            self.cyan_start_counter = 0
            # Reset cyan fallback when exiting special zone
            self.force_white_only = False
            self.last_cyan_loss_time = None
            self.was_using_cyan = False
        
        return False
    
    def force_exit_waypoint_mode(self):
        """Force exit from waypoint mode when cyan line starts"""
        print(f"\n🚦 CYAN LINE STARTED - Forcing exit from waypoint mode → VISION (median method)")
        self.in_special_zone = False
        self.current_waypoint_idx = 0
        self.cyan_start_counter = 0
        # Reset cyan fallback
        self.force_white_only = False
        self.last_cyan_loss_time = None
        self.was_using_cyan = False
    
    def find_target_waypoint(self, car_x, car_y):
        """Find target waypoint ahead using lookahead distance"""
        waypoints = self.active_waypoints
        if waypoints is None:
            return None, None, 0
        
        closest_dist = float('inf')
        closest_idx = self.current_waypoint_idx
        
        # Find closest waypoint to current position
        for i in range(self.current_waypoint_idx, len(waypoints)):
            wx, wy = waypoints[i]
            dist = np.hypot(wx - car_x, wy - car_y)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        # Look ahead from closest waypoint
        for i in range(closest_idx, len(waypoints)):
            wx, wy = waypoints[i]
            dist = np.hypot(wx - car_x, wy - car_y)
            if dist >= self.lookahead:
                self.current_waypoint_idx = i
                return wx, wy, i
        
        # Return last waypoint if nothing found
        self.current_waypoint_idx = len(waypoints) - 1
        return waypoints[-1][0], waypoints[-1][1], len(waypoints) - 1
    
    def pure_pursuit_steering(self, car_x, car_y, car_yaw, target_x, target_y):
        """Pure Pursuit controller for waypoint following"""
        dx = target_x - car_x
        dy = target_y - car_y
        target_angle = np.arctan2(dy, dx)
        alpha = target_angle - car_yaw
        
        # Normalize angle to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        Ld = np.hypot(dx, dy)
        
        # Avoid division by zero
        if Ld < 0.01:
            return 0.0
        
        steering = np.arctan2(2 * self.waypoint_L * np.sin(alpha), Ld)
        return np.clip(steering, -0.3, 0.3)
    
    def detect_lane_boundaries(self, frame):
        """Detect BOTH cyan (left) and white (right) boundaries with time-based fallback"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect CYAN line
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        
        # Detect WHITE line
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Clean up
        kernel = np.ones((5,5), np.uint8)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI
        h, w = frame.shape[:2]
        roi_start_px = int(h * self.roi_start)
        roi_end_px = int(h * min(self.roi_start + self.roi_height, 1.0))
        
        cyan_roi = cyan_mask[roi_start_px:roi_end_px, :]
        white_roi = white_mask[roi_start_px:roi_end_px, :]
        
        # Find CYAN center
        cyan_center = None
        M_cyan = cv2.moments(cyan_roi)
        if M_cyan['m00'] > 100:
            cx = int(M_cyan['m10'] / M_cyan['m00'])
            cy = int(M_cyan['m01'] / M_cyan['m00']) + roi_start_px
            cyan_center = (cx, cy)
        
        # Find WHITE center
        white_center = None
        M_white = cv2.moments(white_roi)
        if M_white['m00'] > 100:
            cx = int(M_white['m10'] / M_white['m00'])
            cy = int(M_white['m01'] / M_white['m00']) + roi_start_px
            white_center = (cx, cy)
        
        # 🎯 TIME-BASED CYAN FALLBACK LOGIC
        current_time = time.time()
        
        # Check if we lost cyan (was using cyan, now it's gone)
        if self.was_using_cyan and not cyan_center:
            if self.last_cyan_loss_time is None:
                # First time losing cyan
                self.last_cyan_loss_time = current_time
                print(f"\n⚠️  CYAN LOST - Starting timeout timer")
            elif current_time - self.last_cyan_loss_time <= self.cyan_switch_timeout:
                # Within timeout window - cyan flickering detected!
                if not self.force_white_only:
                    print(f"\n🔄 CYAN FLICKERING DETECTED - Switching to WHITE-ONLY mode permanently!")
                    self.force_white_only = True
        
        # Reset if cyan comes back and we're not in forced mode
        if cyan_center and not self.force_white_only:
            self.last_cyan_loss_time = None
            self.was_using_cyan = True
        
        # 🎯 IMPROVED LANE CENTER CALCULATION WITH TIME-BASED FALLBACK
        lane_center = None
        strategy = "NONE"
        
        if self.force_white_only and white_center:
            # FORCED WHITE-ONLY: Ignore cyan permanently due to flickering
            target_x = white_center[0] - self.offset_from_white
            target_y = white_center[1]
            lane_center = (target_x, target_y)
            strategy = "FORCED WHITE-ONLY (cyan unstable)"
            
        elif cyan_center and white_center:
            # BEST: Both edges visible - use true center
            center_x = (cyan_center[0] + white_center[0]) // 2
            center_y = (cyan_center[1] + white_center[1]) // 2
            lane_center = (center_x, center_y)
            strategy = "BOTH (center)"
            self.was_using_cyan = True
            
        elif white_center:
            # GOOD: Only white visible - stay fixed distance from it
            target_x = white_center[0] - self.offset_from_white
            target_y = white_center[1]
            lane_center = (target_x, target_y)
            strategy = "WHITE ONLY (following right edge)"
            self.was_using_cyan = False
            
        elif cyan_center:
            # OK: Only cyan visible - stay fixed distance from it
            target_x = cyan_center[0] + self.offset_from_cyan
            target_y = cyan_center[1]
            lane_center = (target_x, target_y)
            strategy = "CYAN ONLY (following left edge)"
            self.was_using_cyan = True
        
        cyan_pixels = cv2.countNonZero(cyan_roi)
        white_pixels = cv2.countNonZero(white_roi)
        combined_mask = cv2.bitwise_or(cyan_mask, white_mask)
        
        return lane_center, cyan_center, white_center, combined_mask, cyan_pixels, white_pixels, (roi_start_px, roi_end_px), strategy
    
    def calculate_steering(self, lane_center, width):
        """Calculate steering with smooth transitions"""
        if lane_center is None:
            self.previous_error = 0
            return 0.0
        
        center_x = width // 2
        error = (lane_center[0] - center_x) / center_x
        
        derivative = error - self.previous_error
        steering = -self.kp * error - self.kd * derivative
        self.previous_error = error
        
        # Apply rate limiting to prevent sudden changes
        steering = self.apply_steering_rate_limit(steering)
        
        return np.clip(steering, -0.3, 0.3)
    
    def run(self):
        """Main loop"""
        
        # Initialize hardware
        try:
            old_stdin = sys.stdin
            sys.stdin = StringIO("1\n")
            from pal.products.qcar import QCar
            qcar_hw = QCar(readMode=1, frequency=100)
            sys.stdin = old_stdin
            use_hardware = True
            print("✅ Hardware control enabled (QCar1)")
        except Exception as e:
            sys.stdin = old_stdin
            use_hardware = False
            print(f"⚠️  No hardware control")
        
        print("\n🚗 Starting lane follower with traffic sign dataset collection...\n")
        
        # Start the timer for speed transition
        self.start_time = time.time()
        
        frame_count = 0
        detected_count = 0
        lost_count = 0
        
        try:
            while True:
                frame_count += 1
                
                # 🚦 CYCLE TRAFFIC LIGHTS
                self.cycle_traffic_lights()
                
                # Get car position for special zone detection
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
                
                # 🐌 CHECK TIMED SLOW ZONE
                in_timed_slow, time_remaining = self.check_timed_slow_zone(car_x, car_y)
                
                # Calculate current speed based on elapsed time
                elapsed_time = time.time() - self.start_time
                if elapsed_time < self.speed_transition_time:
                    current_base_speed = self.initial_speed
                else:
                    current_base_speed = self.cruising_speed
                
                success, image_data = self.qcar.get_image(camera=QLabsQCar.CAMERA_CSI_FRONT)
                
                if success and image_data is not None:
                    try:
                        img = np.frombuffer(image_data, dtype=np.uint8)
                        img = img.reshape((410, 820, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # 📊 DATA COLLECTION - Save frame
                        if self.collect_dataset and frame_count % self.collection_interval == 0:
                            self.dataset_collector.save_frame(
                                img, 
                                frame_count, 
                                car_x, 
                                car_y, 
                                car_yaw, 
                                self.light_states
                            )
                        
                        # 🔀 HYBRID CONTROL - Check for special zone
                        in_special_zone = self.check_special_zone(car_x, car_y)
                        
                        if in_special_zone:
                            # ========== WAYPOINT MODE (No smoothing - needs fast response) ==========
                            target_x, target_y, wp_idx = self.find_target_waypoint(car_x, car_y)
                            steering = self.pure_pursuit_steering(car_x, car_y, car_yaw, target_x, target_y)
                            
                            # 🚦 CHECK FOR CYAN LINE START
                            result = self.detect_lane_boundaries(img)
                            lane_center_vision, cyan_pos, white_pos, mask, cyan_px, white_px, roi_bounds, strategy = result
                            
                            # STRICT MODE for CURVE12 - ignore cyan detection!
                            if self.active_waypoint_name == "CURVE12":
                                mode = "WAYPOINT"
                                strategy = f"{self.active_waypoint_name} [STRICT] [{wp_idx+1}/{len(self.active_waypoints)}]"
                                lane_center = (410, 300)  # Dummy for visualization
                                mode_color = (255, 0, 255)  # Magenta - STRICT mode
                                current_base_speed = self.special_zone_speed  # Use same speed as ROUND16 (0.045)
                            elif cyan_px > self.cyan_start_threshold:  # Cyan line visible (ROUND16 only)
                                self.cyan_start_counter += 1
                                if self.cyan_start_counter >= self.cyan_start_frames:
                                    # Cyan has started! Force exit waypoint mode
                                    self.force_exit_waypoint_mode()
                                    in_special_zone = False
                                    
                                    # Now use VISION mode with smoothing
                                    lane_center = lane_center_vision
                                    steering = self.calculate_steering(lane_center, img.shape[1])
                                    mode = "VISION"
                                    mode_color = (0, 165, 255) if self.force_white_only else (0, 255, 0)
                                else:
                                    mode = "WAYPOINT"
                                    strategy = f"{self.active_waypoint_name} [{wp_idx+1}/{len(self.active_waypoints)}] - Cyan detected ({self.cyan_start_counter}/{self.cyan_start_frames})"
                                    lane_center = (410, 300)
                                    mode_color = (0, 255, 255)  # Yellow
                                    current_base_speed = self.special_zone_speed
                            else:
                                self.cyan_start_counter = 0
                                mode = "WAYPOINT"
                                strategy = f"{self.active_waypoint_name} [{wp_idx+1}/{len(self.active_waypoints)}] - No cyan"
                                lane_center = (410, 300)
                                mode_color = (255, 165, 0)  # Orange
                                current_base_speed = self.special_zone_speed
                        else:
                            # ========== VISION MODE (With ALL smoothing features) ==========
                            result = self.detect_lane_boundaries(img)
                            lane_center_raw, cyan_pos, white_pos, mask, cyan_px, white_px, roi_bounds, strategy = result
                            
                            # 🎯 APPLY SMOOTH TRANSITION BLENDING
                            lane_center_blended = self.smooth_lane_center_transition(lane_center_raw, strategy)
                            
                            # 🎯 APPLY EMA FILTER
                            lane_center = self.apply_ema_filter(lane_center_blended)
                            
                            # Calculate steering with rate limiting
                            steering = self.calculate_steering(lane_center, img.shape[1])
                            
                            mode = "VISION"
                            mode_color = (0, 165, 255) if self.force_white_only else (0, 255, 0)
                            
                            # Store for next frame
                            self.previous_lane_center = lane_center_raw
                            self.previous_strategy = strategy
                        
                        # 🐌 SPEED CONTROL
                        in_post_curve12_slowdown = False
                        post_curve12_time_remaining = 0.0
                        if self.curve12_completed and self.curve12_completion_time is not None:
                            elapsed = time.time() - self.curve12_completion_time
                            if elapsed < self.post_curve12_slowdown_duration:
                                in_post_curve12_slowdown = True
                                post_curve12_time_remaining = self.post_curve12_slowdown_duration - elapsed
                            else:
                                self.curve12_completion_time = None
                                self.curve12_completed = False  # Reset for next lap!
                        
                        # Speed priority
                        if lane_center or in_special_zone:
                            detected_count += 1
                            lost_count = 0
                            
                            if self.active_waypoint_name == "CURVE12":
                                current_speed = self.special_zone_speed  # Use same speed as ROUND16 (0.045)
                                speed_reason = f"{self.active_waypoint_name} STRICT (0.045)"
                            elif in_post_curve12_slowdown:
                                current_speed = self.initial_speed
                                speed_reason = f"POST-CURVE12 ({post_curve12_time_remaining:.1f}s)"
                            elif in_timed_slow:
                                current_speed = self.slow_zone_speed
                                speed_reason = f"SLOW ZONE ({time_remaining:.1f}s)"
                            else:
                                current_speed = current_base_speed
                                speed_reason = "NORMAL"
                        else:
                            lost_count += 1
                            if lost_count < 20:
                                current_speed = current_base_speed * 0.5
                                speed_reason = "SEARCHING"
                            else:
                                current_speed = 0
                                speed_reason = "STOPPED"
                            
                            if lost_count > 100:
                                print(f"\n⚠️  Lost lane!")
                                break
                        
                        # Control
                        if use_hardware:
                            qcar_hw.write(current_speed, steering)
                        
                        # Visualization
                        vis = img.copy()
                        h, w = vis.shape[:2]
                        
                        # Mode indicator
                        mode_text = f"MODE: {mode}"
                        if self.transition_counter > 0:
                            mode_text += f" [SMOOTHING {self.transition_counter}/{self.transition_frames}]"
                        elif self.force_white_only:
                            mode_text += " [WHITE-ONLY LOCKED]"
                        cv2.putText(vis, mode_text, (10, h-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
                        
                        if mode == "VISION":
                            roi_start_px, roi_end_px = roi_bounds
                            roi_color = (0, 255, 0) if (cyan_pos and white_pos) else (0, 165, 255) if (cyan_pos or white_pos) else (0, 0, 255)
                            
                            cv2.rectangle(vis, (0, roi_start_px), (w, roi_end_px), roi_color, 3)
                            
                            # Draw CYAN
                            if cyan_pos:
                                cv2.circle(vis, cyan_pos, 10, (255, 255, 0), -1)
                                cv2.circle(vis, cyan_pos, 13, (0, 0, 0), 2)
                                cv2.putText(vis, "CYAN", (cyan_pos[0]-20, cyan_pos[1]-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
                            # Draw WHITE
                            if white_pos:
                                cv2.circle(vis, white_pos, 10, (255, 255, 255), -1)
                                cv2.circle(vis, white_pos, 13, (0, 0, 0), 2)
                                cv2.putText(vis, "WHITE", (white_pos[0]-25, white_pos[1]-15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Draw SMOOTHED TARGET
                            if lane_center:
                                cv2.circle(vis, lane_center, 15, (0, 255, 0), -1)
                                cv2.circle(vis, lane_center, 18, (0, 0, 0), 2)
                                cv2.putText(vis, "TARGET (SMOOTH)", (lane_center[0]-60, lane_center[1]+35),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.line(vis, (w//2, lane_center[1]), lane_center, (0, 255, 0), 3)
                                
                                if cyan_pos and white_pos and not self.force_white_only:
                                    cv2.line(vis, cyan_pos, white_pos, (255, 0, 255), 2)
                        
                        # Center line
                        cv2.line(vis, (w//2, 0), (w//2, h), (255, 0, 0), 2)
                        
                        # Status
                        if in_post_curve12_slowdown:
                            status_text = f"🔄 POST-CURVE12 ({post_curve12_time_remaining:.1f}s)"
                            status_color = (255, 165, 0)
                        elif in_timed_slow and self.active_waypoint_name != "CURVE12":
                            status_text = f"🐌 SLOW ZONE ({time_remaining:.1f}s)"
                            status_color = (0, 255, 255)
                        elif mode == "WAYPOINT" and self.active_waypoint_name == "CURVE12":
                            status_text = f"🔒 {self.active_waypoint_name} STRICT MODE"
                            status_color = (255, 0, 255)
                        elif mode == "WAYPOINT":
                            status_text = "🗺️ SPECIAL ZONE"
                            status_color = (255, 165, 0)
                        elif self.force_white_only:
                            status_text = "🔒 FORCED WHITE-ONLY"
                            status_color = (0, 165, 255)
                        elif cyan_pos and white_pos:
                            status_text = "✅ BOTH EDGES"
                            status_color = (0, 255, 0)
                        elif white_pos:
                            status_text = "⚠️ WHITE ONLY"
                            status_color = (255, 255, 0)
                        elif cyan_pos:
                            status_text = "⚠️ CYAN ONLY"
                            status_color = (0, 165, 255)
                        else:
                            status_text = f"❌ LOST ({lost_count})"
                            status_color = (0, 0, 255)
                        
                        cv2.putText(vis, f"Speed: {current_speed:.3f} ({speed_reason})", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                        cv2.putText(vis, f"Steering: {steering:.2f}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        cv2.putText(vis, status_text, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        cv2.putText(vis, f"Strategy: {strategy}", (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                        
                        if mode == "VISION":
                            cv2.putText(vis, f"Cyan: {cyan_px}px", (10, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(vis, f"White: {white_px}px", (10, 180),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            cv2.putText(vis, f"Pos: ({car_x:.1f}, {car_y:.1f})", (10, 150),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            cv2.putText(vis, f"Cyan: {cyan_px}px", (10, 180),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        detection_pct = (detected_count/frame_count)*100
                        cv2.putText(vis, f"Rate: {detection_pct:.0f}%", (10, 210),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # 📊 DATASET COLLECTION STATUS
                        if self.collect_dataset:
                            saved_count = len(self.dataset_collector.frame_metadata)
                            cv2.putText(vis, f"📊 Dataset: {saved_count} images", (10, 240),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        if in_timed_slow:
                            cv2.putText(vis, f"⏱️ TIME LEFT: {time_remaining:.1f}s", (10, 270),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        elif self.slow_zone_completed:
                            cv2.putText(vis, "✅ Slow Zone Done", (10, 270),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        steer_x = int(w//2 + (steering * w//4))
                        cv2.arrowedLine(vis, (w//2, h-50), (steer_x, h-50), (0, 255, 255), 4)
                        
                        cv2.imshow('Lane Follower - Traffic Dataset Collection 📊', vis)
                        if mode == "VISION":
                            cv2.imshow('Boundaries Mask', mask)
                        
                        if frame_count % 30 == 0:
                            if in_timed_slow:
                                print(f"Frame {frame_count}: 🐌 SLOW ZONE Pos=({car_x:.1f},{car_y:.1f}) Speed={current_speed:.3f}, Time Left={time_remaining:.1f}s")
                            elif mode == "WAYPOINT":
                                cyan_status = f"Cyan={cyan_px}px" if cyan_px < self.cyan_start_threshold else f"Cyan={cyan_px}px [DETECTED {self.cyan_start_counter}/{self.cyan_start_frames}]"
                                print(f"Frame {frame_count}: 🗺️ {mode} Pos=({car_x:.1f},{car_y:.1f}) Speed={current_speed:.3f}, Steer={steering:.2f}, {cyan_status}, {strategy}")
                            else:
                                smooth_indicator = f"[SMOOTH {self.transition_counter}/{self.transition_frames}]" if self.transition_counter > 0 else ""
                                status = "🔒FORCED" if self.force_white_only else "✅BOTH" if (cyan_pos and white_pos) else "⚠️WHITE" if white_pos else "⚠️CYAN" if cyan_pos else "❌"
                                print(f"Frame {frame_count}: {status} {smooth_indicator} Speed={current_speed:.3f}, Steer={steering:.2f}, "
                                      f"Cyan={cyan_px}px, White={white_px}px, Strategy={strategy}")
                        
                    except Exception as e:
                        print(f"⚠️  Error: {e}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            if use_hardware:
                qcar_hw.write(0, 0)
            
            # 📊 FINALIZE DATASET COLLECTION
            if self.collect_dataset:
                self.dataset_collector.finalize()
            
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            if frame_count > 0:
                print(f"\n📊 Driving Statistics:")
                print(f"   Frames: {frame_count}")
                print(f"   Detected: {detected_count} ({(detected_count/frame_count)*100:.1f}%)")
            
            print("\n✅ Program finished!\n")

if __name__ == '__main__':
    follower = LaneCenterFollower()
    follower.run()