# src/dataset.py

class CustomDataset:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __len__(self):
        return 0  # nanti diganti sesuai dataset
