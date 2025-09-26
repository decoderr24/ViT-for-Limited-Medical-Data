# prepare_dataset.py

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
# Path ke file CSV yang sudah lengkap labelnya
CSV_PATH = 'outputs/labelsNew.csv'
# Path ke dataset asli yang berisi folder 01, 02, ...
SOURCE_DATA_DIR = 'Nuclear Cataract Dataset' 
# Path ke folder baru tempat data split akan disimpan
SPLIT_DATA_DIR = 'data' 
# Rasio untuk data validasi (0.2 berarti 20%)
VALIDATION_SPLIT_RATIO = 0.2

# --- SCRIPT UTAMA ---
print("Membaca file metadata labels.csv...")
df = pd.read_csv(CSV_PATH)

# Ekstrak ID pasien dari path gambar
# Misal: dari '01/DER/image_01.png' menjadi '01'
df['patient_id'] = df['image'].apply(lambda x: x.split('/')[0])

# Dapatkan daftar unik semua pasien
unique_patients = df['patient_id'].unique()
print(f"Total pasien ditemukan: {len(unique_patients)}")

# Bagi daftar pasien menjadi train dan validation
train_patients, valid_patients = train_test_split(
    unique_patients,
    test_size=VALIDATION_SPLIT_RATIO,
    random_state=42 # random_state untuk hasil yang bisa direproduksi
)

print(f"Jumlah pasien untuk training: {len(train_patients)}")
print(f"Jumlah pasien untuk validasi: {len(valid_patients)}")

# Fungsi untuk menyalin file
def copy_files(patient_list, target_folder):
    target_path = os.path.join(SPLIT_DATA_DIR, target_folder)
    subset_df = df[df['patient_id'].isin(patient_list)]
    
    print(f"\nMemproses {len(subset_df)} gambar untuk set '{target_folder}'...")

    for index, row in subset_df.iterrows():
        grade = str(row['grade'])
        image_path = row['image']
        
        class_folder = os.path.join(target_path, grade)
        os.makedirs(class_folder, exist_ok=True)
        
        source_file = os.path.join(SOURCE_DATA_DIR, image_path)
        destination_file = os.path.join(class_folder, os.path.basename(image_path))
        
        # Cek apakah file sumber ada sebelum menyalin
        if os.path.exists(source_file):
            shutil.copyfile(source_file, destination_file)
        else:
            print(f"  Peringatan: File sumber tidak ditemukan -> {source_file}")

# Buat folder utama untuk data split
if os.path.exists(SPLIT_DATA_DIR):
    shutil.rmtree(SPLIT_DATA_DIR)
os.makedirs(SPLIT_DATA_DIR)

# Jalankan proses penyalinan
copy_files(train_patients, 'train')
copy_files(valid_patients, 'valid')

print("\n--- Proses splitting per pasien selesai! ---")
print(f"Dataset baru siap di folder '{SPLIT_DATA_DIR}' dengan struktur yang benar.")