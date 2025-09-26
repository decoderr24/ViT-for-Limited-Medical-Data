# generate_csv.py

import os
import pandas as pd

# --- KONFIGURASI ---
# Path ke dataset asli Anda yang berisi folder 01, 02, 03, ...
SOURCE_DATA_DIR = 'data/Nuclear Cataract Database for Biomedical and Machine Learning Applications\Nuclear Cataract Dataset'
# Nama file CSV yang akan dibuat
OUTPUT_CSV_PATH = 'outputs/csv/labels.csv'

# --- SCRIPT UTAMA ---
image_paths = []

print(f"Memindai folder '{SOURCE_DATA_DIR}'...")

# Berjalan melalui setiap folder pasien (01, 02, ...)
for patient_id in sorted(os.listdir(SOURCE_DATA_DIR)):
    patient_path = os.path.join(SOURCE_DATA_DIR, patient_id)
    
    if os.path.isdir(patient_path):
        # Berjalan melalui setiap subfolder mata (DER, IZQ)
        for eye_folder in os.listdir(patient_path):
            eye_path = os.path.join(patient_path, eye_folder)
            
            if os.path.isdir(eye_path):
                # Cari semua file gambar di dalam folder mata
                for filename in os.listdir(eye_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Buat path relatif yang benar, misal: '01/DER/IM000001.png'
                        relative_path = os.path.join(patient_id, eye_folder, filename)
                        image_paths.append(relative_path)

print(f"Total {len(image_paths)} gambar ditemukan.")

# Buat DataFrame pandas
df = pd.DataFrame({'image': image_paths})

# Tambahkan kolom 'grade' dengan nilai placeholder
df['grade'] = '_LABEL_KOSONG_'

# Simpan ke file CSV
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' berhasil dibuat.")
print("Silakan buka file tersebut dan isi kolom 'grade' dengan label yang benar.")