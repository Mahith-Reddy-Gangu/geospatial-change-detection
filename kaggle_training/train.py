import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LevirCDDataset
from model import SiameseUNet
import os

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return BCE + dice_loss

def calculate_f1_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.item()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # In Kaggle, datasets are usually mounted under /kaggle/input
    # Adjust this path based on where you load the LEVIR-CD dataset in your Kaggle notebook
    dataset_path = '/kaggle/input/levir-cd' 
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} not found.")
        print("Please update 'dataset_path' variable in train.py to match your Kaggle dataset mount path.")
        # We'll create dummy directories so the script doesn't instantly crash if testing locally without data
        return
        
    train_dataset = LevirCDDataset(dataset_path, split='train', img_size=256)
    val_dataset = LevirCDDataset(dataset_path, split='val', img_size=256)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = SiameseUNet().to(device)

    # --- RESUME TRAINING LOGIC ---
    weights_path = 'siamese_unet_weights.pth'
    if os.path.exists(weights_path):
        print(f"✅ Found existing weights! Resuming training from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("No existing weights found. Starting fresh training...")
    # -----------------------------

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 20 # Number of epochs to train
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        
        for batch_idx, (imgA, imgB, mask) in enumerate(train_loader):
            imgA, imgB, mask = imgA.to(device), imgB.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgA, imgB)
            
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_f1 += calculate_f1_score(outputs, mask)
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for imgA, imgB, mask in val_loader:
                imgA, imgB, mask = imgA.to(device), imgB.to(device), mask.to(device)
                outputs = model(imgA, imgB)
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                val_f1 += calculate_f1_score(outputs, mask)
                
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        avg_train_f1 = train_f1/len(train_loader)
        avg_val_f1 = val_f1/len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy (F1): {avg_val_f1*100:.2f}%")
        
    print("Training complete. Saving weights...")
    torch.save(model.state_dict(), 'siamese_unet_weights.pth')
    print("Weights saved to siamese_unet_weights.pth")
    print("You can download this .pth file from the Kaggle output and move it to your local project.")

if __name__ == '__main__':
    train()
