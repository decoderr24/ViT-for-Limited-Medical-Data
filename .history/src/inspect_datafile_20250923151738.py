# inspect_datafile.py
import struct
import os

# --- KONFIGURASI ---
# Arahkan ke salah satu DATAFILE yang ada
# Ganti '01/DER' dengan folder lain jika di sana tidak ada DATAFILE
PATH_TO_DATAFILE = os.path.join('Nuclear Cataract Dataset', '01', 'DER', 'DATAFILE')

# --- SCRIPT UTAMA ---
try:
    with open(PATH_TO_DATAFILE, 'rb') as f:
        # Baca seluruh isi file dalam bentuk biner (bytes)
        content_bytes = f.read()

    print(f"--- Menganalisis File: {PATH_TO_DATAFILE} ---")
    print(f"Total ukuran file: {len(content_bytes)} bytes")
    print(f"Isi file dalam bentuk biner (raw bytes):\n{content_bytes}\n")

    # Mencoba membaca 4 byte pertama sebagai integer (seperti sebelumnya)
    if len(content_bytes) >= 4:
        grade_as_integer = struct.unpack('<i', content_bytes[:4])[0]
        print(f"Interpretasi 4 byte pertama sebagai integer: {grade_as_integer}")

    # Mencoba membaca 8 byte pertama sebagai double (angka desimal)
    if len(content_bytes) >= 8:
        grade_as_double = struct.unpack('<d', content_bytes[:8])[0]
        print(f"Interpretasi 8 byte pertama sebagai double: {grade_as_double}")
        # Angka grade biasanya bilangan bulat, jadi kita bisa bulatkan
        print(f"Nilai double yang dibulatkan: {round(grade_as_double)}")


except FileNotFoundError:
    print(f"!!! ERROR: File tidak ditemukan di '{PATH_TO_DATAFILE}'")
    print("Pastikan path dan nama folder sudah benar.")
except Exception as e:
    print(f"Terjadi error: {e}")