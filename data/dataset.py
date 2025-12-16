"""
PyTorch dataset for face image loading with augmentations
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

class FaceDataset(Dataset):
    """Face image dataset for GAN training"""
    
    def __init__(self, 
                 data_dir: str,
                 resolution: int = 256,
                 mirror: bool = True,
                 normalize: bool = True):
        """
        Args:
            data_dir: Directory with processed face images
            resolution: Target image resolution
            mirror: Enable horizontal flip augmentation
            normalize: Normalize to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.mirror = mirror
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(list(self.data_dir.rglob(ext)))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Build transforms
        transform_list = [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
        
        if mirror:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        transform_list.append(transforms.ToTensor())
        
        if normalize:
            transform_list.append(transforms.Normalize([0.5]*3, [0.5]*3))
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and transform image"""
        img_path = self.image_files[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return random valid image on error
            return self.__getitem__(np.random.randint(len(self)))
    
def get_dataloader(data_dir: str,
                   batch_size: int = 16,
                   resolution: int = 256,
                   num_workers: int = 4,
                   mirror: bool = True,
                   shuffle: bool = True,
                   pin_memory: bool = True) -> DataLoader:
    """
    Create dataloader for training
    
    Args:
        data_dir: Path to processed images
        batch_size: Batch size
        resolution: Image resolution
        num_workers: Number of worker processes
        mirror: Enable horizontal flip
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    dataset = FaceDataset(
        data_dir=data_dir,
        resolution=resolution,
        mirror=mirror
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )
    
    return dataloader

def infinite_dataloader(dataloader: DataLoader):
    """Create infinite iterator from dataloader"""
    while True:
        for batch in dataloader:
            yield batch