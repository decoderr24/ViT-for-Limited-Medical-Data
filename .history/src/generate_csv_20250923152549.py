# generate_csv.py (Versi Debug Lanjutan)

import os
import pandas as pd
import struct

# --- KONFIGURASI ---
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
            
            records = content.split(b'*IM\x00h')
            
            for record in records:
                if len(record) < 1644: continue

                filename_bytes = record[808 : 808 + 16]
                filename = filename_bytes.split(b'\x00')[0].decode('ascii').replace('.JPG', '.png')
                
                grade_bytes = record[1636 : 1636 + 8]
                grade_double = struct.unpack('<d', grade_bytes)[0]
                grade = round(grade_double)

                # --- BARIS DEBUG BARU ---
                # Baris ini akan mencetak setiap file dan grade yang ditemukan
                print(f"Ditemukan: file='{filename}', grade mentah={grade_double:.4f}, grade dibulatkan={grade}")
                
                relative_path = os.path.join(patient_id, eye_folder, filename)
                all_image_data.append({'image': relative_path, 'grade': grade})

        except FileNotFoundError:
            print(f"Peringatan: Tidak ada DATAFILE di folder {eye_path}")
        except Exception as e:
            print(f"Error saat memproses folder {eye_path}: {e}")

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

df = pd.DataFrame(all_image_data)
df_filtered = df[df['grade'] != 0]
print(f"Total gambar valid setelah filtering: {len(df_filtered)}")

df_filtered.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")