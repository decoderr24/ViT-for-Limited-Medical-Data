# generate_csv.py (Versi Paling Final dan Tangguh)

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
        
        # Lompati folder ini jika tidak ada DATAFILE
        if not os.path.exists(datafile_path):
            print(f"Peringatan: Tidak ada DATAFILE di folder {eye_path}")
            continue

        try:
            with open(datafile_path, 'rb') as f:
                content = f.read()

            # Cari semua angka desimal (double) di seluruh file biner
            # Ini cara paling tangguh untuk menemukan kandidat grade
            possible_grades = [g[0] for g in struct.iter_unpack('<d', content)]
            
            # Ambil angka valid (bukan nol) sebagai grade
            valid_grades = [round(g) for g in possible_grades if round(g) != 0]
            
            if not valid_grades:
                # Jika tidak ada grade valid yang ditemukan, lompati folder ini
                print(f"Peringatan: Tidak ditemukan grade valid di {datafile_path}")
                continue
            
            # Asumsikan grade yang benar adalah nilai terbesar yang ditemukan
            grade = max(valid_grades)
            
            # Cari semua file gambar di direktori yang sama
            for filename in os.listdir(eye_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.join(patient_id, eye_folder, filename).replace('\\', '/')
                    print(f"Ditemukan: file='{relative_path}', grade={grade}")
                    all_image_data.append({'image': relative_path, 'grade': grade})

        except Exception as e:
            # Jika masih ada error lain, cetak peringatan dan lanjutkan
            print(f"Error saat memproses {datafile_path}: {e}. Melanjutkan...")
            continue

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

# Hanya buat file jika ada data yang ditemukan
if all_image_data:
    df = pd.DataFrame(all_image_data)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")
else:
    print("Tidak ada data valid yang ditemukan untuk dibuat menjadi file CSV.")

print(f"\n--- Selesai ---")
