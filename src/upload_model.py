from huggingface_hub import HfApi

api = HfApi()

# ganti dengan repo kamu di Hugging Face
repo_id = "Decoder24/ViT-for-Limited-Medical-Data"
model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_weights_only.pth"

# Upload ke repo
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_swin_weights_only.pth",
    repo_id=repo_id,
    repo_type="model"
)

print("✅ File berhasil diupload ke Hugging Face Hub!")
