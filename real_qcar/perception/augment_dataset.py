'''augment_dataset.py

Augment images AND masks together
296 images → ~4500 images
'''

import cv2
import numpy as np
import os
from glob import glob

# Paths
DATA_DIR = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')

# Output directories
AUG_IMAGE_DIR = os.path.join(DATA_DIR, 'augmented_images')
AUG_MASK_DIR = os.path.join(DATA_DIR, 'augmented_masks')

os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

print("=" * 60)
print("=== Dataset Augmentation ===")
print("=" * 60)


def adjust_brightness(image, factor):
    """Adjust brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor):
    """Adjust contrast"""
    mean = np.mean(image)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def add_gaussian_noise(image, sigma=25):
    """Add Gaussian noise"""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rotate_image_and_mask(image, mask, angle):
    """Rotate both image and mask by same angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=0, flags=cv2.INTER_NEAREST)
    
    return rotated_image, rotated_mask


def blur_image(image, ksize=3):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def augment_pair(image, mask, base_name):
    """
    Apply augmentations to image-mask pair
    Returns list of (augmented_image, augmented_mask, new_name)
    """
    augmented = []
    
    # 1. Original
    augmented.append((image.copy(), mask.copy(), f'{base_name}_orig'))
    
    # 2. Brightness variations (important for lighting changes!)
    for i, factor in enumerate([0.6, 0.8, 1.2, 1.4]):
        aug_img = adjust_brightness(image, factor)
        augmented.append((aug_img, mask.copy(), f'{base_name}_bright{i}'))
    
    # 3. Contrast variations
    for i, factor in enumerate([0.8, 1.2]):
        aug_img = adjust_contrast(image, factor)
        augmented.append((aug_img, mask.copy(), f'{base_name}_contrast{i}'))
    
    # 4. Rotation (small angles)
    for angle in [-5, -3, 3, 5]:
        aug_img, aug_mask = rotate_image_and_mask(image, mask, angle)
        augmented.append((aug_img, aug_mask, f'{base_name}_rot{angle}'))
    
    # 5. Gaussian noise
    aug_img = add_gaussian_noise(image, sigma=20)
    augmented.append((aug_img, mask.copy(), f'{base_name}_noise'))
    
    # 6. Blur
    aug_img = blur_image(image, ksize=3)
    augmented.append((aug_img, mask.copy(), f'{base_name}_blur'))
    
    # 7. Combined: brightness + rotation
    aug_img = adjust_brightness(image, 0.7)
    aug_img, aug_mask = rotate_image_and_mask(aug_img, mask, 3)
    augmented.append((aug_img, aug_mask, f'{base_name}_combo1'))
    
    aug_img = adjust_brightness(image, 1.3)
    aug_img, aug_mask = rotate_image_and_mask(aug_img, mask, -3)
    augmented.append((aug_img, aug_mask, f'{base_name}_combo2'))
    
    return augmented


def main():
    # Find all mask files
    mask_files = glob(os.path.join(MASK_DIR, '*_mask.png'))
    
    print(f'Found {len(mask_files)} masks')
    
    if len(mask_files) == 0:
        print("ERROR: No masks found!")
        return
    
    total_augmented = 0
    
    for i, mask_path in enumerate(mask_files):
        # Get corresponding image
        mask_name = os.path.basename(mask_path)
        base_name = mask_name.replace('_mask.png', '')
        image_path = os.path.join(IMAGE_DIR, base_name + '.jpg')
        
        if not os.path.exists(image_path):
            continue
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            continue
        
        # Augment
        augmented_pairs = augment_pair(image, mask, base_name)
        
        # Save augmented pairs
        for aug_img, aug_mask, aug_name in augmented_pairs:
            img_save_path = os.path.join(AUG_IMAGE_DIR, f'{aug_name}.jpg')
            cv2.imwrite(img_save_path, aug_img)
            
            mask_save_path = os.path.join(AUG_MASK_DIR, f'{aug_name}_mask.png')
            cv2.imwrite(mask_save_path, aug_mask)
            
            total_augmented += 1
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f'Processed: {i + 1}/{len(mask_files)}')
    
    print(f'\n{"="*60}')
    print(f'AUGMENTATION COMPLETE!')
    print(f'{"="*60}')
    print(f'Original images: {len(mask_files)}')
    print(f'Augmented images: {total_augmented}')
    print(f'Multiplier: {total_augmented / len(mask_files):.1f}x')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()