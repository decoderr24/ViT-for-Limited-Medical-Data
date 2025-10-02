import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from collections import Counter

def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=2):
    # Augmentasi training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(20)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Augmentasi validasi lebih ringan
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=valid_transform)

    # Hitung distribusi class untuk WeightedRandomSampler
    class_counts = Counter([label for _, label in train_dataset.samples])
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, train_dataset.classes, train_dataset
