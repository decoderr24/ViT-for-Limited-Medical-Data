# src/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size):
    """
    Mendefinisikan pipeline augmentasi dan transformasi data.
    Versi ini menggunakan augmentasi yang kuat untuk mengurangi overfitting.
    """
    # Augmentasi agresif untuk data training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Untuk data validasi, tidak ada augmentasi
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, valid_transform

def get_dataloaders(data_dir, batch_size, image_size):
    """
    Membuat DataLoaders untuk training dan validasi.
    """
    train_transform, valid_transform = get_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(f"{data_dir}/valid", transform=valid_transform)
    
    # DataLoader untuk validasi tetap standar
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, # Sesuaikan dengan kemampuan CPU
        pin_memory=True
    )

    # DataLoader untuk training tidak dibuat di sini lagi,
    # karena akan dibuat di train.py menggunakan sampler.
    # Kita hanya perlu mengembalikan dataset-nya.
    
    dataset_classes = train_dataset.classes
    
    print(f"Data berhasil dimuat. Ditemukan {len(dataset_classes)} kelas: {dataset_classes}")
    
    # Mengembalikan dataset latih agar bisa dibuat sampler-nya di train.py
    return None, valid_loader, dataset_classes, train_dataset