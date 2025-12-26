import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random


class NucleiDataset(Dataset):
    """
    Dataset for nuclei segmentation with high-fidelity and low-fidelity image pairs.
    
    Args:
        high_dir: Path to high-fidelity images
        low_dir: Path to low-fidelity (quantized) images  
        mask_dir: Path to binary masks
        img_size: Size to resize images (default 128)
        is_train: Whether this is training data (enables augmentation)
    """
    
    def __init__(self, high_dir, low_dir, mask_dir, img_size=128, is_train=True):
        super(NucleiDataset, self).__init__()
        
        self.high_dir = high_dir
        self.low_dir = low_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(high_dir) if f.endswith('.png')])
        
        print(f"Loaded {len(self.image_files)} images from {high_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Load images
        high_path = os.path.join(self.high_dir, filename)
        low_path = os.path.join(self.low_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        high_img = Image.open(high_path).convert('RGB')
        low_img = Image.open(low_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Resize if needed
        if high_img.size != (self.img_size, self.img_size):
            high_img = TF.resize(high_img, [self.img_size, self.img_size], interpolation=Image.BICUBIC)
            low_img = TF.resize(low_img, [self.img_size, self.img_size], interpolation=Image.BICUBIC)
            mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        
        # Apply augmentations if training
        if self.is_train:
            high_img, low_img, mask = self.augment(high_img, low_img, mask)
        
        # Convert to tensor
        high_tensor = TF.to_tensor(high_img)
        low_tensor = TF.to_tensor(low_img)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0)
        
        # Normalize images to [-1, 1]
        high_tensor = TF.normalize(high_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        low_tensor = TF.normalize(low_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # Convert mask to integer class labels (0 or 1)
        mask_tensor = (mask_tensor > 0).long().squeeze(0)
        
        return {
            'A': high_tensor,      # High-fidelity image
            'B': low_tensor,       # Low-fidelity image
            'L': mask_tensor       # Label/mask
        }
    
    def augment(self, high_img, low_img, mask):
        """
        Apply data augmentation to image pair and mask.
        Augmentations are applied consistently across all three images.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            high_img = TF.hflip(high_img)
            low_img = TF.hflip(low_img)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            high_img = TF.vflip(high_img)
            low_img = TF.vflip(low_img)
            mask = TF.vflip(mask)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            high_img = TF.rotate(high_img, angle)
            low_img = TF.rotate(low_img, angle)
            mask = TF.rotate(mask, angle)
        
        # Random color jitter (only on images, not mask)
        if random.random() > 0.5:
            color_jitter = transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.1
            )
            high_img = color_jitter(high_img)
            low_img = color_jitter(low_img)
        
        # Random Gaussian blur (only on images, not mask)
        if random.random() > 0.5:
            radius = random.uniform(0, 2.0)
            from PIL import ImageFilter
            high_img = high_img.filter(ImageFilter.GaussianBlur(radius=radius))
            low_img = low_img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return high_img, low_img, mask


def get_dataloader(high_dir, low_dir, mask_dir, batch_size=8, img_size=128, 
                   is_train=True, num_workers=4, shuffle=True):
    """
    Create a DataLoader for nuclei segmentation.
    
    Args:
        high_dir: Path to high-fidelity images
        low_dir: Path to low-fidelity images
        mask_dir: Path to masks
        batch_size: Batch size
        img_size: Image size
        is_train: Training mode (enables augmentation)
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader
    """
    dataset = NucleiDataset(
        high_dir=high_dir,
        low_dir=low_dir,
        mask_dir=mask_dir,
        img_size=img_size,
        is_train=is_train
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    BASE_DIR = Path('/Users/haileyreed/Desktop/Honors_Thesis')
    PATCHES_DIR = BASE_DIR / 'Patches'
    
    # Create train dataloader for MoNuSeg
    train_loader = get_dataloader(
        high_dir=PATCHES_DIR / 'MoNuSeg_train_high',
        low_dir=PATCHES_DIR / 'MoNuSeg_train_low',
        mask_dir=PATCHES_DIR / 'MoNuSeg_train_mask',
        batch_size=8,
        img_size=128,
        is_train=True,
        shuffle=True
    )
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"High-fidelity shape: {batch['A'].shape}")
    print(f"Low-fidelity shape: {batch['B'].shape}")
    print(f"Mask shape: {batch['L'].shape}")
