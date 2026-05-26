import os
import shutil
from pathlib import Path
import random
import yaml

class StopSignOrganizer:
    """
    YOLO Dataset Organizer for Stop Signs ONLY
    Stop Signs (class 0)
    """
    
    def __init__(self, stop_folder, output_folder="traffic_signs_yolo"):
        self.stop_folder = Path(stop_folder)
        self.output_folder = Path(output_folder)
        
        # Class name
        self.class_name = 'stop_sign'  # Class 0
        
        print(f"\n📦 Stop Sign Dataset Organizer")
        print(f"   Class 0: stop_sign")
        
    def collect_images_and_labels(self):
        """Collect all images with their labels from stop folder"""
        images = list(self.stop_folder.glob("*.png")) + list(self.stop_folder.glob("*.jpg"))
        labels_dir = self.stop_folder / "labels"
        
        data = []
        missing_labels = 0
        
        for img in images:
            label_file = labels_dir / f"{img.stem}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Not empty
                        data.append((img, label_file))
                    else:
                        missing_labels += 1
            else:
                missing_labels += 1
        
        if missing_labels > 0:
            print(f"   ⚠️  {missing_labels} images without labels (skipped)")
        
        return data
    
    def organize_dataset(self, train_split=0.8):
        """
        Organize dataset:
        1. Collect stop sign images
        2. Split into train (80%) and val (20%)
        3. Copy to YOLO structure
        """
        print("\n" + "="*70)
        print("🛑 ORGANIZING STOP SIGNS DATASET")
        print("="*70)
        
        # Collect stop signs
        print(f"\n📁 Scanning STOP SIGNS folder...")
        print(f"   Path: {self.stop_folder}")
        stop_data = self.collect_images_and_labels()
        print(f"   ✅ Found {len(stop_data)} labeled stop sign images")
        
        if len(stop_data) == 0:
            print("\n❌ No labeled images found!")
            print("   Did you label the images using the annotator?")
            return None
        
        # Shuffle and split
        random.shuffle(stop_data)
        split_idx = int(len(stop_data) * train_split)
        
        train_data = stop_data[:split_idx]
        val_data = stop_data[split_idx:]
        
        print(f"\n✂️  Split:")
        print(f"   Train: {len(train_data)} images ({train_split*100:.0f}%)")
        print(f"   Val:   {len(val_data)} images ({(1-train_split)*100:.0f}%)")
        
        # Create YOLO directory structure
        train_images_dir = self.output_folder / "train" / "images"
        train_labels_dir = self.output_folder / "train" / "labels"
        val_images_dir = self.output_folder / "val" / "images"
        val_labels_dir = self.output_folder / "val" / "labels"
        
        for folder in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Copy train files
        print(f"\n📋 Copying TRAIN files...")
        for idx, (img_path, label_path) in enumerate(train_data):
            # Copy image
            shutil.copy(img_path, train_images_dir / img_path.name)
            
            # Copy label
            shutil.copy(label_path, train_labels_dir / label_path.name)
            
            if (idx + 1) % 50 == 0:
                print(f"   Copied {idx + 1}/{len(train_data)} train files...")
        
        print(f"   ✅ Copied all {len(train_data)} train files")
        
        # Copy val files
        print(f"\n📋 Copying VAL files...")
        for idx, (img_path, label_path) in enumerate(val_data):
            # Copy image
            shutil.copy(img_path, val_images_dir / img_path.name)
            
            # Copy label
            shutil.copy(label_path, val_labels_dir / label_path.name)
            
            if (idx + 1) % 20 == 0:
                print(f"   Copied {idx + 1}/{len(val_data)} val files...")
        
        print(f"   ✅ Copied all {len(val_data)} val files")
        
        # Create dataset.yaml
        self.create_yaml()
        
        print(f"\n{'='*70}")
        print("✅ DATASET ORGANIZED!")
        print(f"{'='*70}")
        print(f"📁 Output location: {self.output_folder.absolute()}")
        print(f"\n📂 Structure:")
        print(f"   {self.output_folder}/")
        print(f"   ├── train/")
        print(f"   │   ├── images/ ({len(train_data)} images)")
        print(f"   │   └── labels/ ({len(train_data)} labels)")
        print(f"   ├── val/")
        print(f"   │   ├── images/ ({len(val_data)} images)")
        print(f"   │   └── labels/ ({len(val_data)} labels)")
        print(f"   └── dataset.yaml")
        print(f"{'='*70}\n")
        
        return self.output_folder
    
    def create_yaml(self):
        """Create YOLO dataset.yaml configuration file"""
        
        yaml_content = {
            'path': str(self.output_folder.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  # Only 1 class
            'names': ['stop_sign']  # Class 0
        }
        
        yaml_path = self.output_folder / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n📄 Created dataset.yaml")
        print(f"   Location: {yaml_path}")
        print(f"\n   Classes:")
        print(f"      0: stop_sign")
        
        return yaml_path

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🛑 STOP SIGN DATASET ORGANIZER")
    print("="*70)
    print("\nThis organizer creates a YOLO dataset from:")
    print("   Class 0: Stop Signs ONLY")
    print("="*70)
    
    # Get stop signs folder
    print("\n📂 STOP SIGNS:")
    print("   Enter path to your stop_signs_augmented folder")
    print("   (Must contain a 'labels/' subfolder with .txt files)")
    stop_path = input("\n   Path: ").strip().strip('"')
    
    if not stop_path:
        print("\n❌ No path provided!")
        exit()
    
    stop_folder = Path(stop_path)
    if not stop_folder.exists():
        print(f"\n❌ Folder not found: {stop_folder}")
        exit()
    
    if not (stop_folder / "labels").exists():
        print(f"\n❌ No labels/ folder found in {stop_folder}")
        print("   Did you label the images using the annotator?")
        exit()
    
    # Output folder
    output_folder = "traffic_signs_yolo"
    
    print(f"\n📁 Output will be saved to: {output_folder}/")
    
    # Organize
    organizer = StopSignOrganizer(stop_folder, output_folder)
    result = organizer.organize_dataset(train_split=0.8)
    
    if result:
        print("\n" + "="*70)
        print("🎯 NEXT STEP: TRAIN THE MODEL")
        print("="*70)
        print("\n1. Make sure Ultralytics YOLO is installed:")
        print("   pip install ultralytics")
        print(f"\n2. Run training script:")
        print(f"   python train_yolo.py")
        print(f"   Choose option 2 (Traffic Signs Model)")
        print("\n   NOTE: Your model will only detect STOP signs")
        print("="*70 + "\n")