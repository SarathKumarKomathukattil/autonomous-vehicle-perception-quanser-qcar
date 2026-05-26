'''convert_labels_to_masks.py

Convert LabelMe JSON to segmentation masks
'''

import json
import numpy as np
import cv2
import os
from glob import glob

# Class mapping
CLASSES = {
    'yellow_line': 1,
    'roundabout': 2,
}

# Paths
INPUT_DIR = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data\images'
OUTPUT_DIR = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data\masks'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_json_to_mask(json_path):
    """Convert single LabelMe JSON to mask"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # Create empty mask (all background = 0)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Draw each polygon
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        
        if label in CLASSES:
            class_id = CLASSES[label]
            cv2.fillPoly(mask, [points], class_id)
    
    return mask


def main():
    # Find all JSON files
    json_files = glob(os.path.join(INPUT_DIR, '*.json'))
    
    print(f'Found {len(json_files)} labeled images')
    
    if len(json_files) == 0:
        print("ERROR: No JSON files found!")
        print(f"Check folder: {INPUT_DIR}")
        return
    
    converted = 0
    for json_path in json_files:
        try:
            # Convert to mask
            mask = convert_json_to_mask(json_path)
            
            # Save mask
            filename = os.path.basename(json_path).replace('.json', '_mask.png')
            mask_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(mask_path, mask)
            
            converted += 1
            if converted % 50 == 0:
                print(f'Converted: {converted}/{len(json_files)}')
                
        except Exception as e:
            print(f'Error converting {json_path}: {e}')
    
    print(f'\n{"="*50}')
    print(f'DONE!')
    print(f'Converted: {converted} masks')
    print(f'Saved to: {OUTPUT_DIR}')
    print(f'{"="*50}')
    
    # Show sample visualization
    if converted > 0:
        # Load a sample
        sample_json = json_files[0]
        sample_mask = convert_json_to_mask(sample_json)
        
        # Load original image
        img_path = sample_json.replace('.json', '.jpg')
        if os.path.exists(img_path):
            original = cv2.imread(img_path)
            
            # Colorize mask
            colored = np.zeros_like(original)
            colored[sample_mask == 0] = [50, 50, 50]      # background - gray
            colored[sample_mask == 1] = [0, 255, 255]     # yellow_line - cyan
            colored[sample_mask == 2] = [0, 165, 255]     # roundabout - orange
            
            # Blend
            blended = cv2.addWeighted(original, 0.5, colored, 0.5, 0)
            
            # Show
            cv2.imshow('Original', original)
            cv2.imshow('Mask (colored)', colored)
            cv2.imshow('Blended', blended)
            print('\nShowing sample - Press any key to close')
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()