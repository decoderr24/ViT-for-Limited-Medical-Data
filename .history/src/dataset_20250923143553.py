# src/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size):
    """
    Mendefinisikan pipeline augmentasi dan transformasi data.
    Sesuai dengan metodologi: rotasi, flip, brightness, contrast, dll.
    """
    # Augmentasi untuk data training untuk membuat model lebih robust
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # Normalisasi menggunakan mean dan std dari ImageNet (praktik standar)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Untuk data validasi, kita tidak melakukan augmentasi, hanya resize dan normalisasi
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, valid_transform

def get_dataloaders(data_dir, batch_size, image_size):
    """
    Membuat DataLoaders untuk training dan validasi.
    
    Args:
        data_dir (str): Path ke direktori data ('../data').
        batch_size (int): Ukuran batch.
        image_size (int): Ukuran gambar (misal: 224).
        
    Returns:
        train_loader, valid_loader, dataset_classes
    """
    train_transform, valid_transform = get_transforms(image_size)
    
    # Membuat dataset dari folder yang sudah di-split
    # datasets.ImageFolder secara otomatis menemukan kelas dari nama subfolder
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(f"{data_dir}/valid", transform=valid_transform)
    
    # Membuat DataLoader
    # **PENTING**: num_workers di sini untuk mengatasi masalah laptop macet.
    # Jika laptop Anda lambat/macet, ubah num_workers menjadi 2, 1, atau 0.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Turunkan nilai ini jika RAM Anda terbatas
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, # Turunkan nilai ini jika RAM Anda terbatas
        pin_memory=True
    )
    
    # Mendapatkan nama kelas dari dataset
    dataset_classes = train_dataset.classes
    
    print(f"Data berhasil dimuat. Ditemukan {len(dataset_classes)} kelas: {dataset_classes}")
    
    return train_loader, valid_loader, dataset_classes