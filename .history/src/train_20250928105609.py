# src/train.py (Versi Final dengan Test-Time Augmentation)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Impor dari file lain dalam proyek
from model import create_model
from dataset import get_dataloaders
from utils import save_model, save_plots, save_confusion_matrix

# --- 1. KONFIGURASI & HYPERPARAMETERS ---
# Ini adalah parameter final yang bisa Anda gunakan
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data'
OUTPUT_DIR = '../outputs'
IMAGE_SIZE = 224      # Coba naikkan ke 384 jika VRAM cukup, jangan lupa turunkan BATCH_SIZE
BATCH_SIZE = 8        # Ukuran batch yang aman untuk VRAM 4GB
NUM_WORKERS = 4       # Optimal untuk CPU 6-core
EPOCHS = 50
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 3e-5
WEIGHT_DECAY = 0.05   # Regularisasi yang sedikit lebih kuat
MODEL_NAME = 'best_model_final_TTA.pth' # Nama file baru untuk hasil final
NUM_CLASSES = 4

# --- 2. FUNGSI TRAINING & VALIDASI ---
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

def validate_one_epoch_tta(model, dataloader, criterion, device):
    """
    Versi validasi yang menggunakan Test-Time Augmentation (TTA).
    Memprediksi gambar asli dan versi flip, lalu merata-ratakan hasilnya.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader), desc="Validating with TTA"):
            images, labels = images.to(device), labels.to(device)
            
            # TTA: Prediksi gambar asli dan versi flip horizontal
            outputs_original = model(images)
            outputs_flipped = model(torch.flip(images, dims=[3])) # flip horizontal
            
            # Rata-ratakan probabilitas hasil prediksi
            outputs_avg = (torch.softmax(outputs_original, dim=1) + torch.softmax(outputs_flipped, dim=1)) / 2
            
            # Hitung loss dari hasil rata-rata
            loss = criterion(outputs_avg, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs_avg, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions.double() / total_samples).item()
    
    print("\n--- Laporan Klasifikasi Validasi (dengan TTA) ---")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)], zero_division=0))
    
    return epoch_loss, epoch_acc, all_labels, all_preds

# --- 3. SCRIPT UTAMA ---
if __name__ == '__main__':
    _, valid_loader, classes, train_dataset = get_dataloaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)

    print("\n--- Menyiapkan Balanced Sampler untuk Training ---")
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Menggunakan CrossEntropyLoss dengan Label Smoothing untuk regularisasi
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    print("Menggunakan CrossEntropyLoss dengan Label Smoothing (0.1).")
    
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
        validate_one_epoch_tta(model, valid_loader, criterion, DEVICE)

    # --- TAHAP 2: UNFREEZE & FINE-TUNE SELURUH MODEL ---
    print("\n--- TAHAP 2: Fine-tuning Seluruh Model ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_finetune = optim.AdamW(model.parameters(), lr=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer_finetune, mode='min', factor=0.2, patience=5)
    print("Menggunakan scheduler ReduceLROnPlateau.")

    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    best_valid_acc = 0.0
    best_labels, best_preds = None, None
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_finetune, criterion, DEVICE)
        valid_loss, valid_acc, valid_labels, valid_preds = validate_one_epoch_tta(model, valid_loader, criterion, DEVICE)
        
        scheduler.step(valid_loss)
        
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
            
    save_plots(history['train_acc'], history['valid_acc'], history['train_loss'], history['valid_loss'], OUTPUT_DIR)
    
    if best_labels and best_preds:
        save_confusion_matrix(best_labels, best_preds, classes, f"{OUTPUT_DIR}/confusion_matrix.png")
    
    print("\n--- Selesai ---")
    print(f"Model terbaik disimpan di {OUTPUT_DIR}/{MODEL_NAME}")