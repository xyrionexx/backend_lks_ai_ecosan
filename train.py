import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import json
import csv
import copy
from collections import Counter
import torchvision.models as models

try:
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 1. Definisi Model AI (Menggunakan Transfer Learning MobileNetV2)
def get_model(num_classes=6):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze seluruh feature extractor
    for param in model.parameters():
        param.requires_grad = False
        
    # Ambil nilai in_features yang benar secara dinamis
    in_features = model.classifier[1].in_features
    # Buat classifier baru (otomatis requires_grad=True)
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def run_training_phase(phase_name, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience_limit, history_list):
    print(f"\n--- Memulai {phase_name} ---")
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")
        
        history_list.append({
            'phase': phase_name,
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        
        # Checkpoint Saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': epoch_val_acc
            }, "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping triggered pada {phase_name}.")
                break
                
    print(f"\n{phase_name} Selesai! Best Val Loss: {best_val_loss:.4f}")
    # Selalu kembalikan bobot terbaik di fase ini
    model.load_state_dict(best_model_wts)
    return model, history_list

def train_model():
    # SET SEED FOR REPRODUCIBILITY
    torch.manual_seed(42)
    
    print("Memulai proses training model tingkat lanjut (Production-Ready)...")
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
    
    if not os.path.exists(dataset_path):
        print(f"\n\033[91mWARNING: Folder '{dataset_path}' tidak ditemukan!\033[0m")
        print("Pastikan Anda sudah mengekstrak TrashNet ke dalam folder 'dataset'.")
        print("Karena mode dummy aktif, kita akan membangkitkan mock model agar aplikasi tetap berjalan.\n")
        
        # Buat mapping dummy
        dummy_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        with open("class_names.json", "w") as f:
            json.dump(dummy_classes, f)
            
        dummy_model = get_model(num_classes=6)
        torch.save(dummy_model.state_dict(), "best_model.pth")
        
        # Export TorchScript
        dummy_model.eval()
        example_input = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(dummy_model, example_input)
        traced_script_module.save("best_model.ptl")
        
        print("Model dummy berhasil dibuat (best_model.pth & best_model.ptl).")
        return

    # 2. Transformasi Gambar (Preprocessing & Augmentasi Real-World)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset & Exception Handling Dummy (Secara default ImageFolder akan melempar error)
    print("Memuat dataset dari:", dataset_path)
    
    # Filter file gambar yang korup
    def is_valid_file(path):
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    # Buat instance ImageFolder terpisah agar transform tidak bocor antar-subset
    train_dataset_full = datasets.ImageFolder(root=dataset_path, transform=train_transform, is_valid_file=is_valid_file)
    val_test_dataset_full = datasets.ImageFolder(root=dataset_path, transform=val_test_transform, is_valid_file=is_valid_file)
    
    # Save Class Mapping untuk Deployment
    with open("class_names.json", "w") as f:
        json.dump(train_dataset_full.classes, f)
    print(f"Disimpan pemetaan kelas: {train_dataset_full.classes}")

    # 4. Dataset Split (70% Train, 15% Val, 15% Test)
    total_len = len(train_dataset_full)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    # Menggunakan index secara eksplisit
    indices = torch.randperm(total_len, generator=torch.Generator().manual_seed(42)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_ds = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_ds = torch.utils.data.Subset(val_test_dataset_full, val_indices)
    test_ds = torch.utils.data.Subset(val_test_dataset_full, test_indices)
    
    print(f"Total data: {total_len} | Train: {train_size} | Val: {val_size} | Test: {test_size}")

    # 5. Penanganan Class Imbalance dengan WeightedRandomSampler
    train_targets = [train_dataset_full.targets[i] for i in train_ds.indices]
    class_counts = Counter(train_targets)
    weights = [1.0 / class_counts[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    # DataLoaders (num_workers=0 atau 2 sesuai OS)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = get_model(num_classes=len(train_dataset_full.classes))
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    history = []
    
    # ==========================================
    # PHASE 1: Klasifikasi Murni (Backbone Frozen)
    # ==========================================
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=2)
    
    model, history = run_training_phase(
        phase_name="Phase 1 (Classifier)", 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer_ft, 
        scheduler=scheduler_ft, 
        epochs=15, 
        patience_limit=5, 
        history_list=history
    )
    
    # ==========================================
    # PHASE 2: Deep Fine-Tuning (Unfreeze Layer Dalam)
    # ==========================================
    print("\n--- Persiapan Phase 2 ---")
    print("Membuka (Unfreeze) 4 blok terakhir dari MobileNetV2 untuk Fine-Tuning dalam...")
    # MobileNetV2 memiliki features dari 0 s/d 18. Kita buka index 14 ke atas.
    for param in model.features[14:].parameters():
        param.requires_grad = True
        
    print(f"Trainable parameters (Phase 2): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Learning rate SANGAT kecil (1e-5) agar tidak merusak pre-trained weights
    optimizer_fine = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler_fine = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fine, mode='min', factor=0.5, patience=2)
    
    model, history = run_training_phase(
        phase_name="Phase 2 (Deep Fine-Tuning)", 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer_fine, 
        scheduler=scheduler_fine, 
        epochs=10, 
        patience_limit=3, 
        history_list=history
    )
    
    # Save History to CSV
    with open('training_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phase', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        writer.writeheader()
        writer.writerows(history)
    print("\nHistori training lengkap disimpan ke 'training_history.csv'")
    print("State model terbaik otomatis sudah di-load ke dalam model (best_model.pth).")
    
    # 7. Evaluasi Test Set & Per-Class Metrics
    print("\n--- Evaluasi pada Test Set ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if HAS_SKLEARN:
        print("\nClassification Report (Confusion Matrix detail):")
        print(classification_report(all_labels, all_preds, target_names=train_dataset_full.classes))
    else:
        correct_test = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        print(f"Test Accuracy: {100 * correct_test / len(all_labels):.2f}% (Install scikit-learn untuk matrix detail)")
        
    # 8. Ekspor ke format Mobile (TorchScript)
    print("\n--- Mengekspor ke Mobile (TorchScript) ---")
    model.eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("best_model.ptl")
    print("Model berhasil diekspor sebagai 'best_model.ptl'. Siap dipasang di Flutter via pytorch_lite!")

if __name__ == "__main__":
    train_model()
