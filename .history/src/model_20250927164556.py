# src/model.py

import timm
import torch.nn as nn

def create_model(num_classes):
    """
    Membuat model Swin Transformer dengan bobot pre-trained.
    """
    # Menggunakan Swin Transformer Tiny, arsitektur yang kuat dan modern.
    model = timm.create_model(
        'swin_tiny_patch4_window7_224.ms_in22k', 
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.4 # Menggunakan dropout yang sudah terbukti bagus
    )
    
    print(f"Model Swin Transformer berhasil dibuat dengan {num_classes} kelas output.")
    
    return model

if __name__ == '__main__':
    # Uji coba cepat untuk memastikan fungsi berjalan
    model = create_model(num_classes=4)
    print(model.head)