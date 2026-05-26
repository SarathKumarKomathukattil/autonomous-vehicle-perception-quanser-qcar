'''train_lane_segmentation.py

Train U-Net with NICE progress display (like YOLO)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from glob import glob
import time
import sys

# === CONFIGURATION ===
NUM_CLASSES = 3
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DATA_DIR = r'C:\Users\kcksa\Documents\Quanser\5_research\pal_utilities\training_data'
AUG_IMAGE_DIR = os.path.join(DATA_DIR, 'augmented_images')
AUG_MASK_DIR = os.path.join(DATA_DIR, 'augmented_masks')
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'lane_segmentation_model.pth')


def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='━'):
    """Print progress bar like YOLO"""
    percent = f"{100 * (iteration / float(total)):.0f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '─' * (length - filled_length)
    print(f'\r{prefix} {bar} {percent}% {suffix}', end='')
    if iteration == total:
        print()


# === DATASET ===
class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.mask_dir = mask_dir
        
        self.valid_pairs = []
        for img_path in self.image_paths:
            base_name = os.path.basename(img_path).replace('.jpg', '')
            mask_path = os.path.join(mask_dir, f'{base_name}_mask.png')
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_path, mask_path))
        
        print(f"Found {len(self.valid_pairs)} image-mask pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.FloatTensor(image), torch.LongTensor(mask)


# === U-NET MODEL ===
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES):
        super().__init__()
        
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        self.bottleneck = UNetBlock(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        self.out = nn.Conv2d(64, n_classes, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# === TRAINING ===
def train():
    print("=" * 60)
    print("=== Lane Segmentation Training ===")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    dataset = LaneDataset(AUG_IMAGE_DIR, AUG_MASK_DIR)
    
    if len(dataset) == 0:
        print("ERROR: No training data found!")
        return
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f'\nTraining samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Batches per epoch: {len(train_loader)}')
    print(f'Epochs: {NUM_EPOCHS}')
    
    model = UNet(n_classes=NUM_CLASSES).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f'\n{"="*60}')
    print('Starting training...')
    print(f'{"="*60}\n')
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # === TRAIN ===
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # PROGRESS BAR
            print_progress_bar(
                batch_idx + 1, 
                len(train_loader), 
                prefix=f'Epoch {epoch+1:2d}/{NUM_EPOCHS}',
                suffix=f'Loss: {loss.item():.4f}'
            )
        
        train_loss /= len(train_loader)
        
        # === VALIDATE ===
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # === EPOCH SUMMARY ===
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved = "  ★ Saved!"
        
        print(f'         Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s{saved}')
    
    total_time = time.time() - start_time
    
    print(f'\n{"="*60}')
    print('TRAINING COMPLETE!')
    print(f'{"="*60}')
    print(f'Total time: {total_time/60:.1f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Model saved to: {MODEL_SAVE_PATH}')
    print(f'{"="*60}')


if __name__ == '__main__':
    train()
