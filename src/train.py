from src.models import build_model
from transforms import train_transform,val_transform
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataloader import SkinDataset
from sklearn.metrics import accuracy_score, f1_score



def train_model(model_name , batch_size = 32,epochs = 20,learning_rate = 0.001):
    metadata = pd.read_csv("HAM10000_metadata.csv")
    
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

    
    

    train_df, val_df = train_test_split(metadata,test_size=0.2,stratify=metadata["dx"],random_state=42)


    train_dataset = SkinDataset(dataframe=train_df,image_folder="HAM10000_images",transform=train_transform)
    val_dataset = SkinDataset(dataframe=val_df,image_folder="HAM10000_images",transform=val_transform)
    
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)


   
    
    model = build_model(model_name=model_name,num_classes=7)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=2)
    best_loss = float("inf")

    early_stopping = EarlyStopping(patience=5)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,DEVICE)

        val_loss, accuracy, macro_f1, weighted_f1 = validate_one_epoch(model,val_loader,criterion,DEVICE)

        scheduler.step(val_loss)
        best_loss = save_best_model(model,val_loss,best_loss)
        early_stopping(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"Macro F1: {macro_f1:.4f} | "
              f"Weighted F1: {weighted_f1:.4f}"
                )

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break


def train_one_epoch(model,train_loader,criterion,optimizer,device):
    model.train()
    
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        

    return  running_loss / len(train_loader)



def validate_one_epoch(model,val_loader,criterion,device):
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

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    return val_loss, accuracy, macro_f1, weighted_f1

def save_best_model(model,val_loss,best_loss,save_path="best_model.pth"):

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


