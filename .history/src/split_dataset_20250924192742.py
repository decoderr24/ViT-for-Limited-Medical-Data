# split_data.py

import os
import shutil
import random
import math

# --- KONFIGURASI ---
# 1. Ganti 'original_dataset' dengan nama folder tempat Anda mengekstrak data Kaggle
SOURCE_DIR = 'data/dataset' 
# 2. Ini adalah folder tujuan tempat 'train' dan 'valid' akan dibuat
TARGET_DIR = 'data/'
# 3. Rasio pembagian (0.8 berarti 80% untuk training)
TRAIN_RATIO = 0.8

# --- SCRIPT UTAMA ---
print(f"Memulai proses split data dari folder '{SOURCE_DIR}'...")

# Hapus folder target jika sudah ada untuk memulai dari awal
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

# Buat struktur folder target (train dan valid)
train_path = os.path.join(TARGET_DIR, 'train')
valid_path = os.path.join(TARGET_DIR, 'valid')
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)

# Dapatkan daftar semua folder kelas di direktori sumber
try:
    class_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    if not class_folders:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"!!! ERROR: Folder '{SOURCE_DIR}' tidak ditemukan atau kosong.")
    print("Pastikan Anda sudah mengekstrak dataset Kaggle ke dalam folder tersebut.")
    exit()

print(f"Ditemukan {len(class_folders)} kelas: {class_folders}")

# Loop melalui setiap folder kelas
for cls in class_folders:
    source_class_path = os.path.join(SOURCE_DIR, cls)
    
    # Buat subfolder kelas di dalam train dan valid
    train_class_path = os.path.join(train_path, cls)
    valid_class_path = os.path.join(valid_path, cls)
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(valid_class_path, exist_ok=True)
    
    # Dapatkan semua file gambar untuk kelas ini
    images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Acak urutan gambar
    random.shuffle(images)
    
    # Hitung titik pembagian
    split_point = math.floor(len(images) * TRAIN_RATIO)
    
    # Bagi daftar gambar menjadi train dan valid
    train_images = images[:split_point]
    valid_images = images[split_point:]
    
    print(f"  Kelas '{cls}': {len(train_images)} train, {len(valid_images)} valid")
    
    # Salin file-file ke folder tujuan
    for img in train_images:
        shutil.copy(os.path.join(source_class_path, img), os.path.join(train_class_path, img))
        
    for img in valid_images:
        shutil.copy(os.path.join(source_class_path, img), os.path.join(valid_class_path, img))

print("\n--- Proses split data selesai! ---")
print(f"Folder '{TARGET_DIR}' dengan struktur 'train' dan 'valid' telah berhasil dibuat.")