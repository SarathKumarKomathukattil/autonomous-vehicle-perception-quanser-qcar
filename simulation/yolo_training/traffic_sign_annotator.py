import cv2
import os
from pathlib import Path

class StopSignAnnotator:
    def __init__(self, image_dir, labels_dir):
        self.image_dir = Path(image_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(exist_ok=True)
        
        # Determine which class based on folder name
        folder_name = self.image_dir.name.lower()
        if 'stop' in folder_name:
            self.current_class = 0  # stop_sign (class 0)
            self.class_name = 'STOP'
            self.class_color = (0, 0, 255)  # RED
            print("📁 STOP SIGNS folder detected - class 0 (STOP)")
        else:
            self.current_class = 0
            self.class_name = 'STOP'
            self.class_color = (0, 0, 255)
            print("⚠️  Defaulting to class 0 (STOP)")
        
        self.current_image = None
        self.current_image_path = None
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        self.image_files.sort()
        self.current_index = 0
        self.drawing = False
        self.start_point = None
        self.boxes = []
        
        print(f"✅ Found {len(self.image_files)} images in {self.image_dir}")
        print(f"🎨 Class: {self.current_class} ({self.class_name})")
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                end_point = (x, y)
                self.add_box(self.start_point, end_point)
                self.drawing = False
                self.start_point = None
                self.draw_image()
    
    def add_box(self, start, end):
        img_h, img_w = self.current_image.shape[:2]
        
        x1, y1 = start
        x2, y2 = end
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            print("Box too small, skipped")
            return
        
        # YOLO format: center_x, center_y, width, height (normalized)
        center_x = ((x1 + x2) / 2) / img_w
        center_y = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        self.boxes.append([self.current_class, center_x, center_y, width, height])
        print(f"✅ Added box #{len(self.boxes)}")
    
    def draw_image(self):
        display_img = self.current_image.copy()
        img_h, img_w = display_img.shape[:2]
        
        # Draw existing boxes
        for box in self.boxes:
            class_id, center_x, center_y, width, height = box
            
            x1 = int((center_x - width/2) * img_w)
            y1 = int((center_y - height/2) * img_h)
            x2 = int((center_x + width/2) * img_w)
            y2 = int((center_y + height/2) * img_h)
            
            cv2.rectangle(display_img, (x1, y1), (x2, y2), self.class_color, 3)
            cv2.putText(display_img, self.class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.class_color, 2)
        
        # Info overlay - TOP RIGHT
        info_y = 25
        cv2.putText(display_img, f"[{self.current_index + 1}/{len(self.image_files)}]", 
                   (img_w - 150, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_img, f"Boxes: {len(self.boxes)}", 
                   (img_w - 150, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.class_color, 2)
        
        # Instructions at BOTTOM
        cv2.rectangle(display_img, (0, img_h - 30), (img_w, img_h), (0, 0, 0), -1)
        cv2.putText(display_img, "MOUSE=Draw | N=Next | R=Undo | SPACE=Skip | Q=Quit", 
                   (10, img_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('🛑 Stop Sign Annotator', display_img)
    
    def save_labels(self):
        if not self.boxes:
            label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
            if label_file.exists():
                label_file.unlink()
            return
            
        label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
        with open(label_file, 'w') as f:
            for box in self.boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        print(f"💾 Saved {len(self.boxes)} labels")
    
    def load_image(self, index):
        if index >= len(self.image_files) or index < 0:
            print("No more images!")
            return False
            
        self.current_index = index
        self.current_image_path = self.image_files[index]
        self.current_image = cv2.imread(str(self.current_image_path))
        
        if self.current_image is None:
            print(f"Could not load: {self.current_image_path}")
            return False
        
        # Resize if needed
        height, width = self.current_image.shape[:2]
        if height > 800 or width > 1200:
            scale = min(1200/width, 800/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.current_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Load existing labels
        label_file = self.labels_dir / f"{self.current_image_path.stem}.txt"
        self.boxes = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        self.boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
        
        print(f"\n📸 [{self.current_index + 1}/{len(self.image_files)}] {self.current_image_path.name} - {len(self.boxes)} boxes")
        self.draw_image()
        return True
    
    def run(self):
        if not self.image_files:
            print("❌ No images found!")
            return
        
        cv2.namedWindow('🛑 Stop Sign Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('🛑 Stop Sign Annotator', self.mouse_callback)
        
        self.load_image(0)
        
        print("\n" + "="*70)
        print(f"🛑 ANNOTATING: {self.class_name}")
        print("="*70)
        print("🖱️  MOUSE: Click & drag to draw box around the STOP SIGN")
        print("   ⚠️  IMPORTANT: Label ONLY the red octagonal sign")
        print("   ⚠️  Don't label the pole, just the sign itself!")
        print("   💡 For NEGATIVE images (no stop sign): Press SPACE to skip")
        print("⏭️  N or D: Next image (saves automatically)")
        print("⏮️  P or A: Previous image")
        print("🗑️  R: Remove last box")
        print("⏩ SPACE: Skip image (no sign visible / negative sample)")
        print("❌ Q or ESC: Quit")
        print("="*70 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                self.save_labels()
                break
            elif key == ord('n') or key == ord('d'):
                self.save_labels()
                if not self.load_image(self.current_index + 1):
                    print("\n✅ All images annotated!")
                    break
            elif key == ord('p') or key == ord('a'):
                self.save_labels()
                self.load_image(self.current_index - 1)
            elif key == ord('r'):
                if self.boxes:
                    self.boxes.pop()
                    self.draw_image()
                    print("🗑️ Removed last box")
            elif key == ord(' '):
                self.save_labels()  # Save empty label for negatives
                print("⏩ Skipped (no sign visible / negative sample)")
                if not self.load_image(self.current_index + 1):
                    print("\n✅ All images done!")
                    break
        
        cv2.destroyAllWindows()
        print(f"\n{'='*70}")
        print("✅ ANNOTATION COMPLETE!")
        print(f"{'='*70}")
        print(f"📁 Labels saved in: {self.labels_dir}")
        print(f"💡 Remember: Negative samples have empty .txt files")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🛑 Stop Sign Annotator")
    print("="*70)
    print("\nThis tool annotates stop signs (class 0)")
    print("\n⚠️  IMPORTANT: Label ONLY the red octagonal sign, NOT the pole!")
    print("⚠️  For negative images (no stop sign): Press SPACE to create empty label\n")
    
    # Get the folder to annotate
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("📂 Enter the path to your stop sign images folder:")
        print("   Example: traffic_signs_dataset/stop_signs_augmented")
        folder_path = input("\nPath: ").strip().strip('"')
    
    image_dir = Path(folder_path)
    
    if not image_dir.exists():
        print(f"\n❌ Folder not found: {image_dir}")
        print("Please check the path!")
        input("\nPress Enter to exit...")
        exit()
    
    # Labels go in same folder as images
    labels_dir = image_dir / "labels"
    
    print(f"\n📁 Images: {image_dir}")
    print(f"📝 Labels: {labels_dir}")
    
    annotator = StopSignAnnotator(image_dir, labels_dir)
    annotator.run()