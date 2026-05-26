from pathlib import Path
import random

val_labels = Path(r"traffic_lights_yolo\labels\val")

# Find a green light label (class 1)
green_labels = []
for label_file in val_labels.glob("*.txt"):
    if 'green' in label_file.name.lower():
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('1 '):  # Class 1
                green_labels.append((label_file.name, content))

print(f"Found {len(green_labels)} green light labels")

if green_labels:
    # Show first 3
    for i, (name, content) in enumerate(green_labels[:3]):
        print(f"\n{name}:")
        print(f"  {content}")
        parts = content.split()
        if len(parts) >= 5:
            print(f"  Class: {parts[0]}")
            print(f"  Box: center_x={parts[1]}, center_y={parts[2]}, width={parts[3]}, height={parts[4]}")
else:
    print("❌ No green light labels found!")