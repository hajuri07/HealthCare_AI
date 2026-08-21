import sys
from pathlib import Path
from torchvision import datasets
project_root = Path.cwd().parent
sys.path.append(str(project_root))
from tqdm import tqdm
from src.models import build_model
from src.transforms import train_transform, val_transform
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataloader import SkinDataset, class_to_idx
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.transfomr_xray import xray_transform


def train_model(model_name, batch_size=32, epochs=20, learning_rate=0.001):
    print("Kamm dandha chalu hogaya bitwaa")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    IMAGE_FOLDER = r"C:\Ibrahim\Projects\Healthcare\Pytorch-Skin-Diseases-classification\HAM10000_images"
    METADATA = PROJECT_ROOT / "HAM10000_metadata.csv"
    metadata = pd.read_csv(METADATA)

    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

    # split by lesion_id first, so no lesion's photos leak across train/val/test
    unique_lesions = metadata.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]]

    train_lesions, temp_lesions = train_test_split(
        unique_lesions, test_size=0.3, stratify=unique_lesions["dx"], random_state=42
    )
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=0.5, stratify=temp_lesions["dx"], random_state=42
    )

    train_df = metadata[metadata["lesion_id"].isin(train_lesions["lesion_id"])].reset_index(drop=True)
    val_df = metadata[metadata["lesion_id"].isin(val_lesions["lesion_id"])].reset_index(drop=True)
    test_df = metadata[metadata["lesion_id"].isin(test_lesions["lesion_id"])].reset_index(drop=True)

    test_split_path = PROJECT_ROOT / "test_split.csv"
    if not test_split_path.exists():
        test_df.to_csv(test_split_path, index=False)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(7), y=train_df["dx"].map(class_to_idx)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    train_dataset = SkinDataset(dataframe=train_df, image_folder=str(IMAGE_FOLDER), transform=train_transform)
    val_dataset = SkinDataset(dataframe=val_df, image_folder=str(IMAGE_FOLDER), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2, pin_memory= True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(model_name=model_name, num_classes=7).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    best_loss = float("inf")
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, epochs)
        val_loss, accuracy, macro_f1, weighted_f1 = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        best_loss = save_best_model(model, val_loss, best_loss)
        early_stopping(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"Macro F1: {macro_f1:.4f} | "
              f"Weighted F1: {weighted_f1:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=0, total_epochs=1):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}] - Train")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())   # shows live loss next to the bar

    return running_loss / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return val_loss, accuracy, macro_f1, weighted_f1


def save_best_model(model, val_loss, best_loss, save_path="best_model.pth"):
    if val_loss < best_loss:
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved! Validation Loss: {val_loss:.4f}")
        return val_loss
    return best_loss


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_xray_model(model_name, batch_size=32, epochs=20, learning_rate=0.001):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(str(PROJECT_ROOT / "train"), transform=xray_transform)
    val_dataset = datasets.ImageFolder(str(PROJECT_ROOT / "val"), transform=xray_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory= True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory= True)

    model = build_model(model_name=model_name, num_classes=2).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    best_loss = float("inf")
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, accuracy, macro_f1, weighted_f1 = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)
        best_loss = save_best_model(model, val_loss, best_loss)
        early_stopping(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"Macro F1: {macro_f1:.4f} | "
              f"Weighted F1: {weighted_f1:.4f}")

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break