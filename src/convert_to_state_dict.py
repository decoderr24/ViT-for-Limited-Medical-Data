import torch
from torch.serialization import add_safe_globals
from torch.nn.modules.loss import CrossEntropyLoss

# Tambahkan whitelist agar bisa load aman
add_safe_globals([CrossEntropyLoss])

old_model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_model_final.pth"
new_model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_weights_only.pth"

# Load dengan weights_only=False karena ini file trusted
checkpoint = torch.load(old_model_path, map_location="cpu", weights_only=False)

# Deteksi isi file
if isinstance(checkpoint, dict):
    # Coba ambil beberapa kemungkinan key umum
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
        state_dict = checkpoint["model"].state_dict()
    else:
        raise ValueError(f"❌ Tidak ditemukan key state_dict dalam checkpoint: {checkpoint.keys()}")
elif hasattr(checkpoint, "state_dict"):
    # Kalau model langsung
    state_dict = checkpoint.state_dict()
else:
    raise ValueError("❌ File tidak berisi model atau dictionary dengan state_dict yang valid.")

# Simpan ulang hanya weight-nya
torch.save(state_dict, new_model_path)

print(f"✅ State dict berhasil diekstrak dan disimpan ke:\n{new_model_path}")
