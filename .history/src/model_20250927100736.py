# src/model.py

import timm
import torch.nn as nn

def create_model(num_classes):
    """
    Membuat model Vision Transformer dengan bobot pre-trained dan dropout.
    """
    # Menggunakan model ViT yang sama seperti sebelumnya
    # Anda bisa mencoba model lain seperti 'vit_tiny_patch16_224.augreg_in21k'
    model = timm.create_model(
        'vit_base_patch16_224.mae', 
        pretrained=True,
        num_classes=num_classes, # Langsung set jumlah kelas di sini
        drop_rate=0.4 # Menambahkan dropout dengan rate 30%
    )
    
    print(f"Model ViT-MAE berhasil dibuat dengan {num_classes} kelas output dan dropout rate 0.3.")
    
    return model

if __name__ == '__main__':
    # Uji coba cepat untuk memastikan fungsi berjalan
    model = create_model(num_classes=4)
    print(model.head)