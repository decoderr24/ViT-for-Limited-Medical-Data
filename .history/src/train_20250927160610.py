# src/train.py (Versi Final dengan Balanced Sampler & Focal Loss)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import save_model, save_plots, save_confusion_matrix, FocalLoss

# Impor dari file lain dalam proyek
from model import create_model
from dataset import get_dataloaders
from utils import save_model, save_plots, save_confusion_matrix, FocalLoss # Pastikan FocalLoss diimpor

# --- 1. KONFIGURASI & HYPERPARAMETERS ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs/models'
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 3e-5
WEIGHT_DECAY = 0.01
MODEL_NAME = 'best_vit_model_balanced.pth'
NUM_CLASSES = 4

# --- 2. FUNGSI TRAINING & VALIDASI (Tidak ada perubahan di sini) ---
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
    all_preds, all_labels = [], []

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
    
    print("\n--- Laporan Klasifikasi Validasi ---")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)], zero_division=0))
    
    return epoch_loss, epoch_acc, all_labels, all_preds

# --- 3. SCRIPT UTAMA ---
if __name__ == '__main__':
    # A. Persiapan Data & Implementasi WeightedRandomSampler
    # Panggil get_dataloaders untuk mendapatkan dataset latih
    _, valid_loader, classes, train_dataset = get_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

    print("\n--- Menyiapkan Balanced Sampler untuk Training ---")
    # Hitung bobot untuk setiap sampel (bukan per kelas)
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]
    criterion = FocalLoss()
    # Buat Sampler yang akan menyeimbangkan pengambilan data
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Buat DataLoader training BARU yang menggunakan sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4 # Sesuaikan dengan kemampuan CPU Anda
    )
    print("Balanced Sampler berhasil dibuat. Setiap batch training sekarang akan seimbang.")
    
    # B. Arsitektur & Pretraining
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # C. Menggunakan Focal Loss
    # Ganti dari CrossEntropyLoss menjadi FocalLoss
    criterion = FocalLoss(alpha=[1, 1, 1, 2], gamma=2).to(DEVICE)
    print("Menggunakan Focal Loss untuk fokus pada sampel yang sulit.")
    
    # D. Strategi Fine-tune
    # --- TAHAP 1: FREEZE BACKBONE, LATIH HEAD ---
    print("\n--- TAHAP 1: Melatih Classifier Head ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
        
    optimizer_head = optim.AdamW(model.head.parameters(), lr=LEARNING_RATE_HEAD, weight_decay=WEIGHT_DECAY)
    
    for epoch in range(5):
        print(f"Epoch Head {epoch+1}/5")
        train_one_epoch(model, train_loader, optimizer_head, criterion, DEVICE)
        validate_one_epoch(model, valid_loader, criterion, DEVICE)

    # --- TAHAP 2: UNFREEZE & FINE-TUNE SELURUH MODEL ---
    print("\n--- TAHAP 2: Fine-tuning Seluruh Model ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_finetune = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, T_max=EPOCHS, eta_min=1e-7)

    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    best_valid_acc = 0.0
    best_labels, best_preds = None, None
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_finetune, criterion, DEVICE)
        valid_loss, valid_acc, valid_labels, valid_preds = validate_one_epoch(model, valid_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        if valid_acc > best_valid_acc:
            print(f"Validasi akurasi meningkat dari {best_valid_acc:.4f} ke {valid_acc:.4f}. Menyimpan model...")
            save_model(epoch, model, optimizer_finetune, criterion, f"{OUTPUT_DIR}/{MODEL_NAME}")
            best_valid_acc = valid_acc
            best_labels = valid_labels
            best_preds = valid_preds
            
    # E. Evaluasi & Reporting
    save_plots(history['train_acc'], history['valid_acc'], history['train_loss'], history['valid_loss'], OUTPUT_DIR)
    
    if best_labels and best_preds:
        save_confusion_matrix(best_labels, best_preds, classes, f"{OUTPUT_DIR}/confusion_matrix.png")
    
    print("\n--- Selesai ---")
    print(f"Model terbaik disimpan di {OUTPUT_DIR}/{MODEL_NAME}")