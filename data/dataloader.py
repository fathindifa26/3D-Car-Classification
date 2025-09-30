import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import random
import torchvision.transforms.functional as F


class MultiLabelCarDataset(Dataset):
    """
    Dataset class for multi-label car door classification
    """
    
    def __init__(
        self, 
        csv_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_column: str = 'image_path',
        label_columns: List[str] = None
    ):
        """
        Args:
            csv_file: Path to CSV file with labels
            image_dir: Directory containing images
            transform: Optional transform to be applied on images
            image_column: Name of column containing image paths
            label_columns: List of label column names
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_column = image_column
        
        # Default label columns for car doors
        if label_columns is None:
            self.label_columns = [
                'front_left', 'front_right', 'rear_left', 'rear_right', 'hood'
            ]
        else:
            self.label_columns = label_columns
            
        # Verify label columns exist
        missing_cols = set(self.label_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing label columns: {missing_cols}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            labels: Tensor of shape (num_classes,) with binary labels
        """
        # Get image path
        if self.image_column in self.df.columns:
            image_name = self.df.iloc[idx][self.image_column]
        else:
            # Assume first column is image path if not specified
            image_name = self.df.iloc[idx, 0]
            
        image_path = self.image_dir / image_name
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get labels
        labels = []
        for col in self.label_columns:
            label = self.df.iloc[idx][col]
            # Ensure binary label (0 or 1)
            labels.append(float(label) if not pd.isna(label) else 0.0)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance
        Returns weights for each class
        """
        label_counts = self.df[self.label_columns].sum().values
        total_samples = len(self.df)
        
        # Calculate positive class weights
        pos_weights = []
        for count in label_counts:
            if count > 0:
                # Weight = neg_samples / pos_samples
                neg_count = total_samples - count
                weight = neg_count / count
            else:
                weight = 1.0
            pos_weights.append(weight)
        
        return torch.tensor(pos_weights, dtype=torch.float32)
    
    def get_label_distribution(self) -> dict:
        """Get distribution of labels for analysis"""
        distribution = {}
        for col in self.label_columns:
            distribution[col] = {
                'positive': int(self.df[col].sum()),
                'negative': int(len(self.df) - self.df[col].sum()),
                'ratio': float(self.df[col].mean())
            }
        return distribution
class RandomZoomTransform:
    """Custom transform untuk zoom in/out yang lebih kontrol"""
    
    def __init__(self, zoom_range=(0.3, 1.5), output_size=224, p=0.5):
        self.zoom_range = zoom_range
        self.output_size = output_size
        self.p = p
    
    def __call__(self, img):
        # FIX: Correct probability logic
        if random.random() > self.p:
            # No zoom, just resize to output size
            return F.resize(img, (self.output_size, self.output_size))
        
        # Random zoom factor
        zoom_factor = random.uniform(*self.zoom_range)
        
        if zoom_factor > 1.0:
            # ZOOM IN: resize bigger then center crop
            new_size = int(self.output_size * zoom_factor)
            img = F.resize(img, (new_size, new_size))
            img = F.center_crop(img, self.output_size)
        else:
            # ZOOM OUT: resize smaller then pad to center
            new_size = int(self.output_size * zoom_factor)
            img = F.resize(img, (new_size, new_size))
            
            # Calculate padding to center the small car
            pad = (self.output_size - new_size) // 2
            
            # Pad with black color (0)
            img = F.pad(img, pad, fill=0, padding_mode='constant')
            
            # Ensure exact output size
            if img.size != (self.output_size, self.output_size):
                img = F.resize(img, (self.output_size, self.output_size))
        
        return img


def get_transforms(image_size: int = 224, augment: bool = True) -> dict:
    """Get data transforms for training and validation"""
    
    # Base transforms for validation
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Training transforms with augmentation
    if augment:
        train_transforms = transforms.Compose([
            # === PRIMARY: ZOOM AUGMENTATION ===
            RandomZoomTransform(
                zoom_range=(0.3, 1.5),  # 30% zoom out to 150% zoom in
                output_size=image_size,
                p=0.6  # 60% chance to apply zoom
            ),
            
            # === SECONDARY: GEOMETRIC TRANSFORMS ===
            # Small perspective transformation - simulate 3D viewing angle changes
            transforms.RandomPerspective(distortion_scale=0.06, p=0.25),
            
            # Slight affine transformation - simulate camera position variance
            transforms.RandomAffine(
                degrees=0,  # No rotation - handled by 3D scene
                translate=(0.02, 0.02),  # Small translation
                scale=None,  # Scale handled by zoom transform
                shear=None   # No shear for 3D scene
            ),
            
            # === FINAL: NORMALIZATION ===
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transforms = transforms.Compose(base_transforms)
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose(base_transforms)
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }

def create_data_loaders(
    train_csv: str,
    val_csv: str,
    image_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, MultiLabelCarDataset]:
    """
    Create training and validation data loaders
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        image_dir: Directory containing images
        batch_size: Batch size for data loaders
        image_size: Target image size
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset)
    """
    
    # Get transforms
    transforms_dict = get_transforms(image_size, augment)
    
    # Create datasets
    train_dataset = MultiLabelCarDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=transforms_dict['train']
    )
    
    val_dataset = MultiLabelCarDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=transforms_dict['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset