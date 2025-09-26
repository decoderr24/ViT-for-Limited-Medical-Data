# src/train.py
from src import CustomDataset, get_vit_model

if __name__ == "__main__":
    # Coba inisialisasi dataset
    dataset = CustomDataset("data/")
    print("Dataset length:", len(dataset))

    # Coba load model
    model = get_vit_model(num_classes=3)
    print(model)
