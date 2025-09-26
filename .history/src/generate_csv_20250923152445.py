# generate_csv.py (Versi Final untuk membaca metadata kompleks)

import os
import pandas as pd
import struct

# --- KONFIGURASI ---
# Pastikan path ini benar, dijalankan dari folder utama proyek
SOURCE_DATA_DIR = os.path.join(
    'data',
    'Nuclear Cataract Database for Biomedical and Machine Learning Applications',
    'Nuclear Cataract Dataset'
)
OUTPUT_CSV_PATH = 'labels.csv'

# --- SCRIPT UTAMA ---
all_image_data = []

print(f"Memindai folder '{SOURCE_DATA_DIR}'...")

for patient_id in sorted(os.listdir(SOURCE_DATA_DIR)):
    patient_path = os.path.join(SOURCE_DATA_DIR, patient_id)
    if not os.path.isdir(patient_path): continue

    for eye_folder in os.listdir(patient_path):
        eye_path = os.path.join(patient_path, eye_folder)
        if not os.path.isdir(eye_path): continue

        datafile_path = os.path.join(eye_path, 'DATAFILE')
        
        try:
            with open(datafile_path, 'rb') as f:
                content = f.read()
            
            # Cari posisi awal metadata untuk setiap gambar
            # Setiap record metadata diawali dengan b'*IM\x00h'
            records = content.split(b'*IM\x00h')
            
            for record in records:
                if len(record) < 1644: continue # Skip header atau record yang tidak lengkap

                # Ekstrak nama file dari record
                # Posisi nama file (misal: IM000001.JPG) ada di offset 808
                filename_bytes = record[808 : 808 + 16]
                filename = filename_bytes.split(b'\x00')[0].decode('ascii').replace('.JPG', '.png')
                
                # Ekstrak grade dari record
                # Posisi grade (disimpan sebagai double) ada di offset 1636
                grade_bytes = record[1636 : 1636 + 8]
                grade_double = struct.unpack('<d', grade_bytes)[0]
                grade = round(grade_double)
                
                # Buat path relatif
                relative_path = os.path.join(patient_id, eye_folder, filename)
                all_image_data.append({'image': relative_path, 'grade': grade})

        except FileNotFoundError:
            print(f"Peringatan: Tidak ada DATAFILE di folder {eye_path}")
        except Exception as e:
            print(f"Error saat memproses folder {eye_path}: {e}")

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

df = pd.DataFrame(all_image_data)
# Hapus baris di mana grade adalah 0 (jika ada data yang tidak valid)
df = df[df['grade'] != 0]
print(f"Total gambar valid setelah filtering: {len(df)}")


df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")