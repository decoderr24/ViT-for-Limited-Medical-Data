import torch
from torch.serialization import add_safe_globals
from torch.nn.modules.loss import CrossEntropyLoss

# Path model lama (yang masih berisi objek model penuh)
old_model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_model_final.pth"

# Path output baru (hanya berisi bobot/weight)
new_model_path = r"C:\Users\user\Documents\Project\Cataract-ViT\outputs\models\best_swin_weights_only.pth"

# Tambahkan ke daftar yang aman
add_safe_globals([CrossEntropyLoss])

# Load model lama (pastikan ini aman karena model dari kamu sendiri)
model = torch.load(old_model_path, map_location="cpu", weights_only=False)

# Jika model disimpan sebagai dictionary (misal {'model': ..., 'optimizer': ...})
# kita ambil bagian state_dict-nya
if isinstance(model, dict) and "state_dict" in model:
    state_dict = model["state_dict"]
elif hasattr(model, "state_dict"):
    state_dict = model.state_dict()
else:
    raise ValueError("File tidak berisi objek model yang bisa diambil state_dict-nya")

# Simpan ulang hanya state_dict
torch.save(state_dict, new_model_path)

print(f"✅ State dict berhasil disimpan ulang ke:\n{new_model_path}")
