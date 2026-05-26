import cv2
import numpy as np
import time
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime
import pal.resources.rtmodels as rtmodels

"""
STEP 2: INTERACTIVE ROI ADJUSTER
=================================
Adjust ROI (Region of Interest) with trackbars to see which part
of the road to focus on for lane detection.

ROI determines:
- roi_start: How far UP from bottom to start looking (0.0 = bottom, 1.0 = top)
- roi_end: How far UP from bottom to end looking
"""

class ROIAdjuster:
    def __init__(self):
        print("\n" + "="*70)
        print("STEP 2: INTERACTIVE ROI ADJUSTER")
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
        
        # HSV VALUES - Use your tuned values!
        self.lower_road = np.array([0, 1, 5])
        self.upper_road = np.array([174, 18, 203])
        self.lower_cyan = np.array([80, 0, 150])
        self.upper_cyan = np.array([170, 255, 255])
        
        # ROI SETTINGS - Starting values (we'll adjust these)
        self.roi_start = 60  # 60% from bottom (0-100 scale for trackbar)
        self.roi_end = 95    # 95% from bottom
        
        # Center extraction settings
        self.num_points = 10
        
        # Create window with trackbars
        cv2.namedWindow('ROI Adjuster')
        
        # ROI trackbars (0-100 scale, will convert to 0.0-1.0)
        cv2.createTrackbar('ROI Start %', 'ROI Adjuster', self.roi_start, 100, self.on_roi_start)
        cv2.createTrackbar('ROI End %', 'ROI Adjuster', self.roi_end, 100, self.on_roi_end)
        cv2.createTrackbar('Num Points', 'ROI Adjuster', self.num_points, 20, self.on_num_points)
        
        print("\n📏 INTERACTIVE ROI ADJUSTMENT")
        print("="*70)
        print("USE TRACKBARS TO ADJUST:")
        print("  - ROI Start %: Where to START looking (from bottom)")
        print("                 0% = very bottom, 100% = very top")
        print("  - ROI End %:   Where to STOP looking")
        print("  - Num Points:  How many sample points to extract")
        print("\n🎯 GOAL: ROI should cover the road ahead")
        print("         - Too LOW: Only sees road right in front")
        print("         - Too HIGH: Sees too far ahead (less responsive)")
        print("         - GOOD: Covers 30-40% of image height")
        print("\n💡 TIP: Start = 50-70%, End = 90-95%")
        print("\n🎮 Press 'Q' to finish and see final values")
        print("="*70 + "\n")
    
    def on_roi_start(self, val):
        self.roi_start = val
        # Ensure start < end
        if self.roi_start >= self.roi_end:
            self.roi_start = self.roi_end - 5
            cv2.setTrackbarPos('ROI Start %', 'ROI Adjuster', self.roi_start)
    
    def on_roi_end(self, val):
        self.roi_end = val
        # Ensure end > start
        if self.roi_end <= self.roi_start:
            self.roi_end = self.roi_start + 5
            cv2.setTrackbarPos('ROI End %', 'ROI Adjuster', self.roi_end)
    
    def on_num_points(self, val):
        if val < 3:
            val = 3
            cv2.setTrackbarPos('Num Points', 'ROI Adjuster', val)
        self.num_points = val
    
    def create_road_mask(self, frame):
        """Create road mask"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.lower_road, self.upper_road)
        
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def create_cyan_mask(self, frame):
        """Create cyan line mask"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
        
        kernel = np.ones((5, 5), np.uint8)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_OPEN, kernel)
        
        return cyan_mask
    
    def extract_sample_points(self, road_mask):
        """Extract sample points from ROI"""
        h, w = road_mask.shape
        
        # Convert percentages to pixels
        roi_start_frac = self.roi_start / 100.0
        roi_end_frac = self.roi_end / 100.0
        
        roi_start_px = int(h * roi_start_frac)
        roi_end_px = int(h * roi_end_frac)
        
        road_roi = road_mask[roi_start_px:roi_end_px, :]
        
        roi_height = roi_end_px - roi_start_px
        sample_rows = np.linspace(0, roi_height - 1, self.num_points, dtype=int)
        
        sample_points = []
        
        for row_idx in sample_rows:
            road_row = road_roi[row_idx, :]
            road_pixels = np.where(road_row > 0)[0]
            
            if len(road_pixels) > 5:
                center_x = int(np.mean(road_pixels))
                center_y = row_idx + roi_start_px
                sample_points.append((center_x, center_y))
        
        return sample_points, roi_start_px, roi_end_px
    
    def run(self):
        """Main adjustment loop"""
        print("\n📏 STARTING ROI ADJUSTER...\n")
        
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
                        
                        # Create masks
                        road_mask = self.create_road_mask(img)
                        cyan_mask = self.create_cyan_mask(img)
                        
                        # Extract sample points
                        sample_points, roi_start_px, roi_end_px = self.extract_sample_points(road_mask)
                        
                        # VISUALIZATION 1: Original with ROI box and sample points
                        vis_orig = img.copy()
                        
                        # Draw ROI box (GREEN = active region)
                        cv2.rectangle(vis_orig, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 3)
                        
                        # Draw sample points
                        for pt in sample_points:
                            cv2.circle(vis_orig, pt, 8, (0, 0, 255), -1)  # Red dots
                        
                        # Draw horizontal lines at sample rows
                        roi_height = roi_end_px - roi_start_px
                        sample_rows = np.linspace(0, roi_height - 1, self.num_points, dtype=int)
                        for row_idx in sample_rows:
                            y = row_idx + roi_start_px
                            cv2.line(vis_orig, (0, y), (w, y), (255, 0, 255), 1)  # Magenta lines
                        
                        # Show current ROI values
                        roi_text = f"ROI: {self.roi_start}% to {self.roi_end}% | Points: {len(sample_points)}/{self.num_points}"
                        cv2.putText(vis_orig, roi_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        roi_height_pct = self.roi_end - self.roi_start
                        coverage_text = f"ROI Coverage: {roi_height_pct}% of image"
                        cv2.putText(vis_orig, coverage_text, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Status
                        if 25 <= roi_height_pct <= 45:
                            status = "GOOD Coverage!"
                            status_color = (0, 255, 0)
                        elif roi_height_pct < 25:
                            status = "Too narrow - increase range"
                            status_color = (255, 165, 0)
                        else:
                            status = "Too wide - may see too far"
                            status_color = (255, 165, 0)
                        
                        cv2.putText(vis_orig, status, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        
                        # VISUALIZATION 2: Masks with ROI overlay
                        road_mask_rgb = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
                        cyan_mask_rgb = cv2.cvtColor(cyan_mask, cv2.COLOR_GRAY2BGR)
                        
                        # Draw ROI on masks
                        cv2.rectangle(road_mask_rgb, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 3)
                        cv2.rectangle(cyan_mask_rgb, (0, roi_start_px), (w, roi_end_px), (0, 255, 0), 3)
                        
                        # Draw sample points on masks
                        for pt in sample_points:
                            cv2.circle(road_mask_rgb, pt, 8, (0, 0, 255), -1)
                            cv2.circle(cyan_mask_rgb, pt, 8, (0, 0, 255), -1)
                        
                        # Combine visualizations
                        top_row = cv2.hconcat([vis_orig, vis_orig])  # Original twice for layout
                        bottom_row = cv2.hconcat([road_mask_rgb, cyan_mask_rgb])
                        combined = cv2.vconcat([top_row, bottom_row])
                        
                        # Resize for display
                        display_height = 800
                        aspect_ratio = combined.shape[1] / combined.shape[0]
                        display_width = int(display_height * aspect_ratio)
                        combined_resized = cv2.resize(combined, (display_width, display_height))
                        
                        cv2.imshow('ROI Adjuster', combined_resized)
                        
                        # Console output
                        if frame_count % 30 == 0:
                            roi_start_frac = self.roi_start / 100.0
                            roi_end_frac = self.roi_end / 100.0
                            
                            print(f"\nFrame {frame_count}:")
                            print(f"  ROI Start: {self.roi_start}% ({roi_start_frac:.2f})")
                            print(f"  ROI End: {self.roi_end}% ({roi_end_frac:.2f})")
                            print(f"  Coverage: {roi_height_pct}% of image")
                            print(f"  Sample points: {len(sample_points)}/{self.num_points}")
                            
                            if 25 <= roi_height_pct <= 45:
                                print(f"  ✅ Good coverage!")
                            else:
                                print(f"  ⚠️ Adjust coverage")
                        
                    except Exception as e:
                        if frame_count % 100 == 0:
                            print(f"⚠️  Processing error: {e}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping adjuster!")
                    break
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        
        finally:
            cv2.destroyAllWindows()
            QLabsRealTime().terminate_all_real_time_models()
            
            # Print final values
            roi_start_frac = self.roi_start / 100.0
            roi_end_frac = self.roi_end / 100.0
            
            print("\n" + "="*70)
            print("✅ FINAL ROI VALUES:")
            print("="*70)
            print(f"self.roi_start = {roi_start_frac:.2f}  # {self.roi_start}%")
            print(f"self.roi_end = {roi_end_frac:.2f}  # {self.roi_end}%")
            print(f"self.num_points = {self.num_points}")
            print("="*70)
            print("\n💡 Copy these values into your main code!")
            print("   Recommended coverage: 25-45% of image height\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STEP 2: INTERACTIVE ROI ADJUSTMENT")
    print("="*70)
    print("\n🎯 GOAL: Define which part of road to look at")
    print("\n📝 HOW TO USE:")
    print("   1. Window shows:")
    print("      - Green box: ROI region (where we look)")
    print("      - Red dots: Sample points extracted")
    print("      - Magenta lines: Sample row positions")
    print("\n   2. Adjust trackbars:")
    print("      - ROI Start %: How far UP from bottom to start")
    print("                     (50-70% recommended)")
    print("      - ROI End %:   Where to stop looking")
    print("                     (90-95% recommended)")
    print("      - Num Points:  How many samples (8-12 good)")
    print("\n   3. Good ROI covers:")
    print("      ✅ Enough road ahead to plan steering")
    print("      ✅ Not so far that distant road dominates")
    print("      ✅ 25-45% of image height")
    print("\n   4. Press 'Q' when satisfied")
    print("   5. Copy values into your main code")
    print("="*70 + "\n")
    
    input("Press ENTER to start ROI adjustment... ")
    
    adjuster = ROIAdjuster()
    adjuster.run()