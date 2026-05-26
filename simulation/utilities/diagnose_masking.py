import cv2
import numpy as np
import time
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

"""
STEP 1: INTERACTIVE HSV TUNING TOOL
====================================
Adjust HSV values with trackbars until mask looks perfect!

GOAL: Road mask should show ONLY the dark road surface
      - NOT sidewalks
      - NOT shoulders
      - NOT background
"""

class HSVTuner:
    def __init__(self):
        print("\n" + "="*70)
        print("STEP 1: INTERACTIVE HSV TUNING TOOL")
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
        
        # STARTING VALUES (we'll adjust these with trackbars)
        self.lower_h = 0
        self.lower_s = 0
        self.lower_v = 30
        
        self.upper_h = 180
        self.upper_s = 40
        self.upper_v = 100
        
        # ROI
        self.roi_start = 0.60
        self.roi_end = 0.95
        
        # Create window with trackbars
        cv2.namedWindow('HSV Tuner')
        
        # Lower HSV trackbars
        cv2.createTrackbar('Lower H', 'HSV Tuner', self.lower_h, 180, self.on_lower_h)
        cv2.createTrackbar('Lower S', 'HSV Tuner', self.lower_s, 255, self.on_lower_s)
        cv2.createTrackbar('Lower V', 'HSV Tuner', self.lower_v, 255, self.on_lower_v)
        
        # Upper HSV trackbars
        cv2.createTrackbar('Upper H', 'HSV Tuner', self.upper_h, 180, self.on_upper_h)
        cv2.createTrackbar('Upper S', 'HSV Tuner', self.upper_s, 255, self.on_upper_s)
        cv2.createTrackbar('Upper V', 'HSV Tuner', self.upper_v, 255, self.on_upper_v)
        
        print("\n🎨 INTERACTIVE HSV TUNING")
        print("="*70)
        print("USE TRACKBARS TO ADJUST:")
        print("  - Lower V (Value): Darkest road color to detect")
        print("  - Upper V (Value): Brightest road color to detect")
        print("  - Upper S (Saturation): Max color intensity")
        print("\n🎯 GOAL: White mask should ONLY show dark road")
        print("         - NOT bright areas (sidewalks)")
        print("         - NOT left/right edges unless they're road")
        print("\n🎮 Press 'Q' to finish and see final values")
        print("="*70 + "\n")
    
    def on_lower_h(self, val):
        self.lower_h = val
    
    def on_lower_s(self, val):
        self.lower_s = val
    
    def on_lower_v(self, val):
        self.lower_v = val
    
    def on_upper_h(self, val):
        self.upper_h = val
    
    def on_upper_s(self, val):
        self.upper_s = val
    
    def on_upper_v(self, val):
        self.upper_v = val
    
    def create_road_mask(self, frame):
        """Create road mask with CURRENT trackbar values"""
        lower_road = np.array([self.lower_h, self.lower_s, self.lower_v])
        upper_road = np.array([self.upper_h, self.upper_s, self.upper_v])
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def analyze_mask(self, road_mask):
        """Get statistics from mask"""
        h, w = road_mask.shape
        
        roi_start_px = int(h * self.roi_start)
        roi_end_px = int(h * self.roi_end)
        road_roi = road_mask[roi_start_px:roi_end_px, :]
        
        # Sample middle row
        middle_row_idx = road_roi.shape[0] // 2
        middle_row = road_roi[middle_row_idx, :]
        
        road_pixels = np.where(middle_row > 0)[0]
        
        if len(road_pixels) > 10:
            left_edge = np.min(road_pixels)
            right_edge = np.max(road_pixels)
            road_width = right_edge - left_edge
            return left_edge, right_edge, road_width
        
        return None, None, None
    
    def run(self):
        """Main tuning loop"""
        print("\n🎨 STARTING HSV TUNER...\n")
        
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
                        
                        h, w = img.shape[:2]
                        roi_start_px = int(h * self.roi_start)
                        roi_end_px = int(h * self.roi_end)
                        
                        # Create mask with current values
                        road_mask = self.create_road_mask(img)
                        
                        # Analyze
                        left_edge, right_edge, road_width = self.analyze_mask(road_mask)
                        
                        # VISUALIZATION 1: Original with ROI
                        vis_orig = img.copy()
                        cv2.rectangle(vis_orig, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 2)
                        
                        # Show current HSV values on image
                        lower_text = f"Lower: [{self.lower_h}, {self.lower_s}, {self.lower_v}]"
                        upper_text = f"Upper: [{self.upper_h}, {self.upper_s}, {self.upper_v}]"
                        cv2.putText(vis_orig, lower_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(vis_orig, upper_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # VISUALIZATION 2: Mask overlay on original
                        vis_overlay = img.copy()
                        
                        # Show road mask in green (semi-transparent)
                        road_color = np.zeros_like(img)
                        road_color[:, :, 1] = road_mask  # Green channel
                        vis_overlay = cv2.addWeighted(vis_overlay, 0.7, road_color, 0.3, 0)
                        
                        # Draw edge lines if available
                        if left_edge is not None and right_edge is not None:
                            cv2.line(vis_overlay, (left_edge, roi_start_px), 
                                    (left_edge, roi_end_px), (255, 0, 0), 3)  # Blue = left
                            cv2.line(vis_overlay, (right_edge, roi_start_px), 
                                    (right_edge, roi_end_px), (0, 0, 255), 3)  # Red = right
                            
                            # Show width
                            cv2.putText(vis_overlay, f"Width: {road_width}px", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            # Status
                            if 200 <= road_width <= 450:
                                status = "GOOD WIDTH!"
                                status_color = (0, 255, 0)
                            elif road_width > 450:
                                status = "TOO WIDE - Detecting sidewalk?"
                                status_color = (0, 0, 255)
                            else:
                                status = "TOO NARROW"
                                status_color = (255, 165, 0)
                            
                            cv2.putText(vis_overlay, status, (10, 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        
                        cv2.rectangle(vis_overlay, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 2)
                        
                        # VISUALIZATION 3: Pure mask
                        road_mask_rgb = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(road_mask_rgb, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 2)
                        
                        # Draw edges on mask
                        if left_edge is not None and right_edge is not None:
                            cv2.line(road_mask_rgb, (left_edge, roi_start_px), 
                                    (left_edge, roi_end_px), (255, 0, 0), 3)
                            cv2.line(road_mask_rgb, (right_edge, roi_start_px), 
                                    (right_edge, roi_end_px), (0, 0, 255), 3)
                        
                        # Combine visualizations
                        top_row = cv2.hconcat([vis_orig, vis_overlay])
                        bottom_row = cv2.hconcat([road_mask_rgb, road_mask_rgb])  # Duplicate for layout
                        combined = cv2.vconcat([top_row, bottom_row])
                        
                        # Resize for display
                        display_height = 800
                        aspect_ratio = combined.shape[1] / combined.shape[0]
                        display_width = int(display_height * aspect_ratio)
                        combined_resized = cv2.resize(combined, (display_width, display_height))
                        
                        cv2.imshow('HSV Tuner', combined_resized)
                        
                        # Console output
                        if frame_count % 30 == 0:
                            road_pixels_total = cv2.countNonZero(road_mask)
                            
                            print(f"\nFrame {frame_count}:")
                            print(f"  HSV: [{self.lower_h},{self.lower_s},{self.lower_v}] to [{self.upper_h},{self.upper_s},{self.upper_v}]")
                            print(f"  Road pixels: {road_pixels_total}")
                            
                            if left_edge is not None and right_edge is not None:
                                print(f"  Left edge: {left_edge}")
                                print(f"  Right edge: {right_edge}")
                                print(f"  Width: {road_width}px", end="")
                                
                                if 200 <= road_width <= 450:
                                    print(" ✅ GOOD!")
                                elif road_width > 450:
                                    print(" ❌ TOO WIDE!")
                                else:
                                    print(" ⚠️ TOO NARROW")
                        
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"⚠️  Processing error: {e}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping tuner!")
                    break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            # Print final values
            print("\n" + "="*70)
            print("✅ FINAL HSV VALUES:")
            print("="*70)
            print(f"self.lower_road = np.array([{self.lower_h}, {self.lower_s}, {self.lower_v}])")
            print(f"self.upper_road = np.array([{self.upper_h}, {self.upper_s}, {self.upper_v}])")
            print("="*70)
            print("\n💡 Copy these values into your main code!")
            print("   They should give you a good road mask.\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STEP 1: INTERACTIVE HSV TUNING")
    print("="*70)
    print("\n🎯 GOAL: Find HSV values that detect ONLY the road")
    print("\n📝 HOW TO USE:")
    print("   1. Window will show 4 views:")
    print("      - Top-left: Original + HSV values")
    print("      - Top-right: Green overlay (detected road)")
    print("      - Bottom: Mask (white = detected)")
    print("\n   2. Adjust trackbars until:")
    print("      ✅ Mask shows ONLY dark road (diagonal strip)")
    print("      ✅ Width: 200-450px")
    print("      ❌ NO white on left/right edges (unless road)")
    print("      ❌ NO sidewalk/shoulder detection")
    print("\n   3. Press 'Q' when satisfied")
    print("   4. Copy final values into your main code")
    print("\n💡 TIP: Start with Upper V (Value) - lower it until")
    print("        bright areas (sidewalks) disappear!")
    print("="*70 + "\n")
    
    input("Press ENTER to start interactive tuning... ")
    
    tuner = HSVTuner()
    tuner.run()