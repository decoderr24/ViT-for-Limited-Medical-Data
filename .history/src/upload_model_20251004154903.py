from huggingface_hub import upload_file

# === 1. Ganti path lokal model ===
local_model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_model_final.pth"

# === 2. Ganti repo_id sesuai nama repo kamu ===
repo_id = "Decoder24/ViT-for-Limited-Medical-Data"

# === 3. Upload file model ===
upload_file(
    path_or_fileobj=local_model_path,
    path_in_repo="best_swin_model_final.pth",  # nama file di repo Hugging Face
    repo_id=repo_id,
    repo_type="model"
)

print("✅ Model berhasil diupload ke Hugging Face Hub!")
