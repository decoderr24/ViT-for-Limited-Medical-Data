# src/utils.py

import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

# Atur style plot agar terlihat lebih bagus
plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion, model_path):
    """
    Fungsi untuk menyimpan checkpoint model.
    """
    print(f"Menyimpan model ke {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        }, model_path)

def save_plots(train_acc, valid_acc, train_loss, valid_loss, plot_path):
    """
    Fungsi untuk menyimpan plot akurasi dan loss selama training.
    """
    print(f"Menyimpan plot ke {plot_path}")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # Plot Akurasi
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{plot_path}/accuracy.png")
    
    # Plot Loss
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{plot_path}/loss.png")

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Menyimpan plot confusion matrix untuk analisis kesalahan model.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Best Validation Model')
    plt.savefig(save_path)
    print(f"Confusion matrix disimpan di {save_path}")