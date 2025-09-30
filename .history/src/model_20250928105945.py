# src/model.py

import timm
import torch.nn as nn

def create_model(num_classes, image_size=224):
    """
    Membuat model Swin Transformer dengan ukuran input yang bisa disesuaikan.
    """
    model = timm.create_model(
        'swin_tiny_patch4_window7_224.ms_in22k', 
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.4,
        img_size=image_size # Tambahkan parameter ini untuk menyesuaikan ukuran input
    )
    
    print(f"Model Swin Transformer berhasil dibuat untuk input {image_size}x{image_size}.")
    
    return model