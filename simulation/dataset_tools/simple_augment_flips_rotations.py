import cv2
import numpy as np
import os
from pathlib import Path

class StopSignAugmentor:
    """
    Simple augmentation for stop signs
    ONLY flips and small rotations - perfect for simulation
    Augments 230 images (180 stop + 50 negatives) to 1000 total
    """
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(self.input_dir.glob(ext)))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} original images")
    
    def flip_horizontal(self, img):
        """Flip horizontally"""
        return cv2.flip(img, 1)
    
    def rotate_image(self, img, angle):
        """Rotate image by angle (degrees)"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def adjust_brightness(self, img, factor):
        """Adjust brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def augment_to_target(self, target_total=1000):
        """
        Augment dataset to reach target total
        Uses flips, rotations, and brightness adjustments
        """
        original_count = len(self.image_files)
        
        print(f"\n{'='*70}")
        print(f"🛑 STOP SIGN AUGMENTATION")
        print(f"{'='*70}")
        print(f"   Original images: {original_count}")
        print(f"   Target total: {target_total}")
        print(f"   Need to create: {target_total - original_count}")
        print(f"   Methods: Flip + Rotations + Brightness")
        print(f"{'='*70}\n")
        
        # Copy original images first
        print("Step 1: Copying original images...")
        for img_file in self.image_files:
            img = cv2.imread(str(img_file))
            output_path = self.output_dir / img_file.name
            cv2.imwrite(str(output_path), img)
        print(f"✅ Copied {original_count} original images")
        
        # Calculate how many augmented versions we need
        needed = target_total - original_count
        
        if needed <= 0:
            print(f"\n✅ Already have enough images!")
            return
        
        print(f"\nStep 2: Creating {needed} augmented images...")
        
        # Define augmentations (12 types)
        augmentations = [
            ('flip', self.flip_horizontal),
            ('rot_left_2', lambda img: self.rotate_image(img, -2)),
            ('rot_right_2', lambda img: self.rotate_image(img, 2)),
            ('rot_left_3', lambda img: self.rotate_image(img, -3)),
            ('rot_right_3', lambda img: self.rotate_image(img, 3)),
            ('rot_left_5', lambda img: self.rotate_image(img, -5)),
            ('rot_right_5', lambda img: self.rotate_image(img, 5)),
            ('rot_left_7', lambda img: self.rotate_image(img, -7)),
            ('rot_right_7', lambda img: self.rotate_image(img, 7)),
            ('bright_0.8', lambda img: self.adjust_brightness(img, 0.8)),
            ('bright_1.2', lambda img: self.adjust_brightness(img, 1.2)),
            ('bright_0.7', lambda img: self.adjust_brightness(img, 0.7)),
        ]
        
        total_created = 0
        aug_counter = 0
        
        # Keep cycling through images and augmentations until we reach target
        while total_created < needed:
            for img_file in self.image_files:
                if total_created >= needed:
                    break
                
                img = cv2.imread(str(img_file))
                
                # Pick augmentation (cycle through them)
                aug_idx = aug_counter % len(augmentations)
                aug_name, aug_func = augmentations[aug_idx]
                
                # Apply augmentation
                aug_img = aug_func(img)
                
                # Save with unique name
                stem = img_file.stem
                ext = img_file.suffix
                aug_filename = f"{stem}_aug{total_created+1}_{aug_name}{ext}"
                output_path = self.output_dir / aug_filename
                cv2.imwrite(str(output_path), aug_img)
                
                total_created += 1
                aug_counter += 1
                
                if total_created % 50 == 0:
                    print(f"   Created {total_created}/{needed} augmented images...")
        
        print(f"\n✅ Created {total_created} augmented images")
        
        # Count final total
        final_images = list(self.output_dir.glob('*.png')) + list(self.output_dir.glob('*.jpg'))
        
        print(f"\n{'='*70}")
        print("📊 SUMMARY")
        print(f"{'='*70}")
        print(f"   Original images: {original_count}")
        print(f"   Augmented images: {total_created}")
        print(f"   TOTAL IMAGES: {len(final_images)}")
        print(f"   Location: {self.output_dir}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🛑 STOP SIGN AUGMENTOR - Target: 1000 images")
    print("="*70)
    print("\nAugmentations used:")
    print("  - Horizontal flip")
    print("  - Small rotations (-7, -5, -3, -2, +2, +3, +5, +7 degrees)")
    print("  - Brightness adjustments (0.7x, 0.8x, 1.2x)")
    print("\nPerfect for simulation environments!")
    print("="*70)
    
    # Get stop signs folder path
    print("\n📂 Enter the path to your stop_signs folder:")
    print("   Example: traffic_signs_dataset/stop_signs")
    stop_input = input("\nPath: ").strip().strip('"')
    
    if not stop_input:
        print("\n❌ No path provided!")
        exit()
    
    input_path = Path(stop_input)
    
    if not input_path.exists():
        print(f"\n❌ Folder not found: {input_path}")
        print("Please check the path!")
        exit()
    
    # Output folder
    output_path = input_path.parent / f"{input_path.name}_augmented"
    
    print(f"\n📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Create augmentor and run
    augmentor = StopSignAugmentor(input_path, output_path)
    augmentor.augment_to_target(target_total=1000)
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output: {output_path}")
    print(f"📊 Total: 1000 stop sign images ready!")
    print("\n🎯 NEXT STEPS:")
    print("   1. Label the 1000 augmented stop sign images (if not already)")
    print("   2. Split into train/val/test sets")
    print("   3. Train your YOLO model!")
    print("   4. Use conf_threshold=0.5 to reduce false positives")
    print("="*70 + "\n")