# src/model.py

import timm
import torch.nn as nn

def create_model(num_classes):
    """
    Membuat model Vision Transformer dengan bobot pre-trained MAE.
    
    Args:
        num_classes (int): Jumlah kelas output.
    
    Returns:
        model: Model PyTorch.
    """
    # Muat model ViT Base dengan patch size 16x16 dan input 224x224
    # 'pretrained=True' akan mengunduh bobot yang sudah di-pre-train di ImageNet dengan metode MAE
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
    
    # Ganti lapisan klasifikasi terakhir (head) agar sesuai dengan jumlah kelas kita
    # 'n_features' akan mendapatkan jumlah fitur input ke lapisan head secara otomatis
    n_features = model.head.in_features
    model.head = nn.Linear(n_features, num_classes)
    
    print(f"Model ViT-MAE berhasil dibuat dengan {num_classes} kelas output.")
    
    return model

if __name__ == '__main__':
    # Uji coba cepat untuk memastikan fungsi berjalan
    model = create_model(num_classes=4) # Misal, 4 kelas katarak
    print(model.head)