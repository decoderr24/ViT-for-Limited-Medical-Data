# generate_csv.py (Versi Final Otomatis)

import os
import pandas as pd

# --- KONFIGURASI ---
# Path ke dataset asli Anda yang berisi folder 01, 02, ...
# Dijalankan dari folder utama proyek (Cataract-ViT)
SOURCE_DATA_DIR = 'data/Nuclear Cataract Database for Biomedical and Machine Learning Applications/Nuclear Cataract Dataset' 
OUTPUT_CSV_PATH = 'outputs/csv/label.csv'

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
                    # Baca grade dari DATAFILE
                    with open(datafile_path, 'r') as f:
                        grade = f.read().strip() # .strip() untuk menghapus spasi/baris baru
                    
                    # Cari semua file gambar di dalam folder mata
                    for filename in os.listdir(eye_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Buat path relatif, misal: '01/DER/IM000001.png'
                            relative_path = os.path.join(patient_id, eye_folder, filename)
                            # Tambahkan data ke daftar
                            all_image_data.append({'image': relative_path, 'grade': grade})
                
                except FileNotFoundError:
                    print(f"Peringatan: Tidak ada DATAFILE di folder {eye_path}")
                except Exception as e:
                    print(f"Error saat memproses folder {eye_path}: {e}")

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

# Buat DataFrame pandas dari daftar data
df = pd.DataFrame(all_image_data)

# Simpan ke file CSV
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")