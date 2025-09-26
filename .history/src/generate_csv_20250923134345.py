# generate_csv.py (Versi Final dengan Pembacaan Biner)

import os
import pandas as pd
import struct

# --- KONFIGURASI ---
SOURCE_DATA_DIR = 'data/Nuclear Cataract Database for Biomedical and Machine Learning Applications/Nuclear Cataract Dataset' 
OUTPUT_CSV_PATH = 'outputs/csv/labelsNew.csv'

# --- SCRIPT UTAMA ---
all_image_data = []

print(f"Memindai folder '{SOURCE_DATA_DIR}' untuk membuat labels.csv secara otomatis...")

for patient_id in sorted(os.listdir(SOURCE_DATA_DIR)):
    patient_path = os.path.join(SOURCE_DATA_DIR, patient_id)
    
    if os.path.isdir(patient_path):
        for eye_folder in os.listdir(patient_path):
            eye_path = os.path.join(patient_path, eye_folder)
            
            if os.path.isdir(eye_path):
                datafile_path = os.path.join(eye_path, 'DATAFILE')
                
                try:
                    # Baca grade dari DATAFILE dalam mode biner ('rb')
                    with open(datafile_path, 'rb') as f:
                        # Asumsikan grade adalah integer pertama dalam file biner
                        grade_bytes = f.read(4) # Baca 4 byte pertama untuk integer
                        grade = struct.unpack('<i', grade_bytes)[0]
                    
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