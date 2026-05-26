from ultralytics import YOLO
import torch
from pathlib import Path

class YOLOTrainer:
    def __init__(self, dataset_yaml, model_name, output_dir):
        self.dataset_yaml = Path(dataset_yaml)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check if CUDA available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
    def train(self, epochs=100, imgsz=640, batch=16):
        """Train YOLO model"""
        print("\n" + "="*70)
        print(f"🚀 TRAINING: {self.model_name}")
        print("="*70)
        print(f"📁 Dataset: {self.dataset_yaml}")
        print(f"📊 Epochs: {epochs}")
        print(f"🖼️  Image size: {imgsz}")
        print(f"📦 Batch size: {batch}")
        print(f"🖥️  Device: {self.device}")
        print("="*70 + "\n")
        
        # Load pretrained YOLOv8 nano model (smallest/fastest)
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=str(self.dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=self.model_name,
            project=str(self.output_dir),
            device=self.device,
            patience=20,  # Early stopping
            save=True,
            plots=True,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        # Get best model path
        best_model = self.output_dir / self.model_name / "weights" / "best.pt"
        print(f"📁 Best model saved at: {best_model}")
        print("="*70 + "\n")
        
        return best_model

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 YOLO MODEL TRAINER")
    print("="*70)
    
    # Ask which model to train
    print("\n🔍 What do you want to train?")
    print("   1 = Traffic Lights Model 🚦 (Red & Green)")
    print("   2 = Traffic Signs Model 🛑 (Stop Signs Only)")
    print("   3 = Both (one after another)")
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        # Train traffic lights model
        print("\n🚦 TRAINING TRAFFIC LIGHTS MODEL")
        print("   Classes: Red Light & Green Light")
        dataset_yaml = "traffic_lights_yolo/dataset.yaml"
        
        if not Path(dataset_yaml).exists():
            print(f"\n❌ Dataset not found: {dataset_yaml}")
            print("   Run the organizer first!")
            exit()
        
        trainer = YOLOTrainer(
            dataset_yaml=dataset_yaml,
            model_name="traffic_light_detector",
            output_dir="traffic_light_training"
        )
        
        best_model = trainer.train(epochs=100, imgsz=640, batch=16)
        
        print("\n🎯 NEXT STEPS:")
        print(f"   Test your model using the combined detector!")
        
    elif choice == '2':
        # Train traffic signs model
        print("\n🛑 TRAINING TRAFFIC SIGNS MODEL")
        print("   Classes: Stop Sign ONLY")
        dataset_yaml = "traffic_signs_yolo/dataset.yaml"
        
        if not Path(dataset_yaml).exists():
            print(f"\n❌ Dataset not found: {dataset_yaml}")
            print("   Run the organizer first!")
            exit()
        
        trainer = YOLOTrainer(
            dataset_yaml=dataset_yaml,
            model_name="traffic_sign_detector",
            output_dir="traffic_sign_training"
        )
        
        best_model = trainer.train(epochs=100, imgsz=640, batch=16)
        
        print("\n🎯 NEXT STEPS:")
        print(f"   Test your model using the combined detector!")
        
    elif choice == '3':
        # Train both models
        print("\n🚀 TRAINING BOTH MODELS")
        print("="*70)
        
        # Train lights first
        print("\n🚦 STEP 1/2: TRAINING TRAFFIC LIGHTS MODEL")
        print("   Classes: Red Light & Green Light")
        dataset_yaml_lights = "traffic_lights_yolo/dataset.yaml"
        
        if Path(dataset_yaml_lights).exists():
            trainer_lights = YOLOTrainer(
                dataset_yaml=dataset_yaml_lights,
                model_name="traffic_light_detector",
                output_dir="traffic_light_training"
            )
            best_model_lights = trainer_lights.train(epochs=100, imgsz=640, batch=16)
        else:
            print(f"⚠️  Skipping lights - dataset not found")
        
        # Train signs second
        print("\n🛑 STEP 2/2: TRAINING TRAFFIC SIGNS MODEL")
        print("   Classes: Stop Sign ONLY")
        dataset_yaml_signs = "traffic_signs_yolo/dataset.yaml"
        
        if Path(dataset_yaml_signs).exists():
            trainer_signs = YOLOTrainer(
                dataset_yaml=dataset_yaml_signs,
                model_name="traffic_sign_detector",
                output_dir="traffic_sign_training"
            )
            best_model_signs = trainer_signs.train(epochs=100, imgsz=640, batch=16)
        else:
            print(f"⚠️  Skipping signs - dataset not found")
        
        print("\n" + "="*70)
        print("🎉 ALL MODELS TRAINED!")
        print("="*70)
        print("\n📁 Model Locations:")
        if Path(dataset_yaml_lights).exists():
            print(f"   🚦 Lights: traffic_light_training/traffic_light_detector/weights/best.pt")
        if Path(dataset_yaml_signs).exists():
            print(f"   🛑 Signs:  traffic_sign_training/traffic_sign_detector/weights/best.pt")
        print("="*70)
        print("\n📝 Your Models:")
        print("   Traffic Lights: Red & Green")
        print("   Traffic Signs: Stop Only (no roundabout)")
        
    else:
        print("\n❌ Invalid choice!")
        exit()
    
    print("\n✅ Training script complete!\n")