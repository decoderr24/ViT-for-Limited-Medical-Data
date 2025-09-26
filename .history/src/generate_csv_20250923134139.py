# generate_csv.py (Versi Final dengan Perbaikan Encoding)

import os
import pandas as pd

# --- KONFIGURASI ---
# Dijalankan dari folder utama proyek (Cataract-ViT)
SOURCE_DATA_DIR = 'Nuclear Cataract Dataset' 
OUTPUT_CSV_PATH = 'labels.csv'

# --- SCRIPT UTAMA ---
all_image_data = []

print(f"Memindai folder '{SOURCE_DATA_DIR}' untuk membuat labels.csv secara otomatis...")

# Berjalan melalui setiap folder pasien (01, 02, ...)
for patient_id in sorted(os.listdir(SOURCE_DATA_DIR)):
    patient_path = os.path.join(SOURCE_DATA_DIR, patient_id)
    
    if os.path.isdir(patient_path):
        # Berjalan melalui setiap subfolder mata (DER, IZQ)
        for eye_folder in os.listdir(patient_path):
            eye_path = os.path.join(patient_path, eye_folder)
            
            if os.path.isdir(eye_path):
                # Path ke file DATAFILE yang berisi grade
                datafile_path = os.path.join(eye_path, 'DATAFILE')
                
                try:
                    # Baca grade dari DATAFILE dengan encoding UTF-8
                    with open(datafile_path, 'r', encoding='utf-8') as f:
                        grade = f.read().strip()
                    
                    # Cari semua file gambar di dalam folder mata
                    for filename in os.listdir(eye_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            relative_path = os.path.join(patient_id, eye_folder, filename)
                            all_image_data.append({'image': relative_path, 'grade': grade})
                
                except FileNotFoundError:
                    print(f"Peringatan: Tidak ada DATAFILE di folder {eye_path}")
                except Exception as e:
                    print(f"Error saat memproses folder {eye_path}: {e}")

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

df = pd.DataFrame(all_image_data)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")