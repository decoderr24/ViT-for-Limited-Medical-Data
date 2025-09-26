# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

# Impor dari file lain dalam proyek
from model import create_model
from dataset import get_dataloaders
from utils import save_model, save_plots

# --- 1. KONFIGURASI & HYPERPARAMETERS ---
# Sebagian besar diambil dari rekomendasi metodologi Anda
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data/split data'  # Pastikan folder 'data' hasil prepare_dataset.py ada di sini
OUTPUT_DIR = '/outputs/models' # Folder untuk menyimpan model dan plot
IMAGE_SIZE = 224  # Ukuran input untuk ViT (bisa 224x224 atau 384x384)
BATCH_SIZE = 16
EPOCHS = 50       # Total epoch, tapi kita akan pakai Early Stopping
LEARNING_RATE_HEAD = 1e-3  # LR untuk melatih classifier head saja
LEARNING_RATE_FINETUNE = 3e-5 # LR untuk fine-tuning seluruh model
WEIGHT_DECAY = 0.01
MODEL_NAME = 'vit_mae_cataract_classifier.pth'
NUM_CLASSES = 4 # Sesuaikan dengan jumlah kelas katarak Anda

# --- 2. FUNGSI UNTUK MENGATASI CLASS IMBALANCE ---
def calculate_class_weights(dataset):
    """
    Menghitung bobot untuk setiap kelas berdasarkan frekuensinya (1 / sqrt(freq)).
    Ini adalah implementasi dari "Weighted loss" di metodologi Anda.
    """
    # Hitung jumlah sampel per kelas
    class_counts = np.bincount(dataset.targets)
    # Hitung bobot: 1 / sqrt(frekuensi)
    class_weights = 1. / np.sqrt(class_counts)
    # Normalisasi bobot
    class_weights = class_weights / np.sum(class_weights)
    return torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# --- 3. FUNGSI TRAINING & VALIDASI ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in tqdm(dataloader, total=len(dataloader), desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    
    # Cetak laporan klasifikasi (precision, recall, f1-score per kelas)
    print("\n--- Laporan Klasifikasi Validasi ---")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)]))
    
    return epoch_loss, epoch_acc

# --- 4. SCRIPT UTAMA ---
if __name__ == '__main__':
    # A. Persiapan & Split Data
    # Kita asumsikan ini sudah dilakukan oleh prepare_dataset.py
    # get_dataloaders akan mengambil data dari folder 'data/train' dan 'data/valid'
    train_loader, valid_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    # B. Arsitektur & Pretraining
    # Menggunakan ViT pretrained (MAE) dari model.py
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # C. Menangani Imbalance
    # Hitung bobot kelas dari dataset training
    class_weights = calculate_class_weights(train_loader.dataset)
    print(f"Class weights untuk mengatasi imbalance: {class_weights}")
    # Gunakan Weighted Cross-Entropy Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # D. Strategi Fine-tune
    # --- TAHAP 1: FREEZE BACKBONE, LATIH HEAD ---
    print("\n--- TAHAP 1: Melatih Classifier Head ---")
    # Bekukan semua layer kecuali head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    optimizer_head = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    
    # Latih head selama beberapa epoch
    for epoch in range(5): # Misal 5 epoch untuk head
        print(f"Epoch Head {epoch+1}/5")
        train_one_epoch(model, train_loader, optimizer_head, criterion, DEVICE)
        validate_one_epoch(model, valid_loader, criterion, DEVICE)

    # --- TAHAP 2: UNFREEZE & FINE-TUNE SELURUH MODEL ---
    print("\n--- TAHAP 2: Fine-tuning Seluruh Model ---")
    # Cairkan semua layer
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_finetune = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY)
    # Gunakan scheduler untuk menyesuaikan learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=EPOCHS, eta_min=1e-7)

    # E. Loop Training Utama dengan Regularisasi & Hyperparams
    history = {
        'train_loss': [], 'train_acc': [],
        'valid_loss': [], 'valid_acc': []
    }
    best_valid_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_finetune, criterion, DEVICE)
        valid_loss, valid_acc = validate_one_epoch(model, valid_loader, criterion, DEVICE)
        
        scheduler.step() # Update learning rate
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # Simpan riwayat
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        # Simpan model dengan akurasi validasi terbaik (Early stopping sederhana)
        if valid_acc > best_valid_acc:
            print(f"Validasi akurasi meningkat dari {best_valid_acc:.4f} ke {valid_acc:.4f}. Menyimpan model...")
            save_model(epoch, model, optimizer_finetune, criterion, f"{OUTPUT_DIR}/{MODEL_NAME}")
            best_valid_acc = valid_acc
            
    # F. Evaluasi & Reporting
    # Simpan plot akurasi dan loss
    save_plots(
        history['train_acc'], history['valid_acc'],
        history['train_loss'], history['valid_loss'],
        OUTPUT_DIR
    )
    
    print("\n--- Selesai ---")
    print(f"Model terbaik disimpan di {OUTPUT_DIR}/{MODEL_NAME}")