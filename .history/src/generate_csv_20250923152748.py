# generate_csv.py (Versi Final dengan Ekstraksi Angka Desimal)

import os
import pandas as pd
import struct
import re

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

# Gunakan os.walk untuk menjelajahi semua subdirektori secara efisien
for root, dirs, files in os.walk(SOURCE_DATA_DIR):
    # Cari DATAFILE di antara file-file yang ada
    if 'DATAFILE' in files:
        datafile_path = os.path.join(root, 'DATAFILE')
        grade = 0 # Default grade jika tidak ditemukan

        try:
            with open(datafile_path, 'rb') as f:
                content = f.read()
            
            # Ekstrak semua angka desimal (float/double) dari file biner
            # Ini akan menemukan semua kemungkinan kandidat grade
            possible_grades = [g for g in struct.iter_unpack('<d', content)]
            
            # Cari kandidat grade yang paling mungkin (bukan nol)
            valid_grades = [round(g[0]) for g in possible_grades if round(g[0]) != 0]
            
            if valid_grades:
                grade = max(valid_grades) # Ambil nilai grade terbesar yang valid
            
            # Sekarang, cari semua file gambar di direktori yang sama
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Buat path relatif dari direktori sumber utama
                    relative_path = os.path.relpath(os.path.join(root, filename), SOURCE_DATA_DIR)
                    # Ganti backslash Windows dengan forward slash untuk konsistensi
                    relative_path = relative_path.replace('\\', '/')
                    
                    print(f"Ditemukan: file='{relative_path}', grade={grade}")
                    all_image_data.append({'image': relative_path, 'grade': grade})

        except Exception as e:
            print(f"Error saat memproses folder {root}: {e}")

print(f"\nTotal {len(all_image_data)} gambar ditemukan dan diberi label.")

df = pd.DataFrame(all_image_data)
# Hapus baris di mana grade adalah 0 (data yang tidak memiliki grade valid)
df_filtered = df[df['grade'] != 0]
print(f"Total gambar valid setelah filtering: {len(df_filtered)}")

df_filtered.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n--- Selesai ---")
print(f"File '{OUTPUT_CSV_PATH}' yang sudah terisi lengkap berhasil dibuat.")