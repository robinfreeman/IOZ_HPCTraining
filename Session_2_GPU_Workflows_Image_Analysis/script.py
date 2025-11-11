# ==============================================================
# GPU Image Classification with ResNet-152
# --------------------------------------------------------------
# Fine-tuning a pretrained ResNet-152 model for bird species
# classification using PyTorch and TorchVision.
# ==============================================================

# ---- Install dependencies (if needed) ----
# pip install torch torchvision tqdm matplotlib

# ---- Imports ----
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import resnet152, ResNet152_Weights

from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# ==============================================================
# 1. GPU Configuration
# ==============================================================

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ==============================================================
# 2. Dataset and Transformations
# ==============================================================

train_path = "/data/torchdatasets/bird-species/train/"
test_path  = "/data/torchdatasets/bird-species/test/"

# ---- Training transformations with augmentation ----
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- Validation/Test transformations (no augmentation) ----
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==============================================================
# 3. Data Loading
# ==============================================================

# Load datasets
full_train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset       = datasets.ImageFolder(root=test_path,  transform=test_transform)

# Split training data into train/validation (80/20 split)
train_size = int(0.8 * len(full_train_dataset))
val_size   = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}\n")


# ==============================================================
# 4. Model Setup
# ==============================================================

# Load pretrained ResNet-152
model = resnet152(weights=ResNet152_Weights.DEFAULT)

# Replace the fully connected layer for our dataset
num_classes = len(full_train_dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Regularisation
    nn.Linear(512, num_classes)
)

# Move model to GPU (if available)
model = model.to(device)

# Unfreeze last two layers for fine-tuning
for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


# ==============================================================
# 5. Optimiser, Loss Function, Scheduler
# ==============================================================

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)


# ==============================================================
# 6. Training Loop
# ==============================================================

num_epochs = 100
train_losses, val_losses = [], []

start_time = time.time()

# Outer progress bar to track epoch progress
with tqdm(total=num_epochs, desc="Overall Training Progress") as epoch_progress:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Show inner tqdm bars only every 20 epochs
        if (epoch + 1) % 20 == 0:
            train_desc = f"Epoch {epoch+1}/{num_epochs} - Training"
            val_desc   = "Validation Loop"
            show_progress = True
        else:
            train_desc = ""
            val_desc   = ""
            show_progress = False

        # ------------------------------
        # Training phase
        # ------------------------------
        for images, labels in tqdm(train_loader, desc=train_desc, disable=not show_progress):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        # Compute mean training loss
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ------------------------------
        # Validation phase
        # ------------------------------
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=val_desc, disable=not show_progress):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Update epoch progress
        epoch_progress.update(1)

        # Print summary every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Step learning rate scheduler
        scheduler.step(val_loss)

        # Optional: tqdm log message
        if (epoch + 1) % 20 == 0:
            tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

end_time = time.time()
training_duration = end_time - start_time


# ==============================================================
# 7. Post-Training: Plot and Save Learning Curves
# ==============================================================

# Create results folder if it doesn’t exist
os.makedirs("results", exist_ok=True)
plot_path_png = "results/training_validation_loss.png"
plot_path_pdf = "results/training_validation_loss.pdf"

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig(plot_path_png, dpi=300)
plt.savefig(plot_path_pdf)
plt.show()

print(f"\n✅ Saved training loss plot to:\n  • {plot_path_png}\n  • {plot_path_pdf}")


# ==============================================================
# 8. GPU Benchmark Summary
# --------------------------------------------------------------
# Display GPU usage and performance information
# ==============================================================

print("\n" + "=" * 60)
print(" GPU BENCHMARK SUMMARY ")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"GPU: {gpu_name}")
    print(f"Total memory: {total_mem:.2f} GB")
    print(f"Allocated memory: {allocated_mem:.2f} GB")
    print(f"Reserved memory: {reserved_mem:.2f} GB")
else:
    print("Running on CPU (no CUDA GPU detected).")

print(f"\nTraining time: {training_duration/60:.2f} minutes")
print("=" * 60 + "\n")

# ============================================================== 
# End of script
# ============================================================== 
