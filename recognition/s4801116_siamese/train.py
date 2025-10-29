import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from dataset import get_data_loaders

def train_epoch(model, loader, criterion, optimizer, device):
    """ Runs one epoch of training. """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (anchor, positive, negative, _) in enumerate(loader):
        anchor, positive, negative = (
            anchor.to(device), 
            positive.to(device), 
            negative.to(device)
        )

        optimizer.zero_grad()
        
        # Forward pass
        anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
        
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    """ Runs one epoch of validation. """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for anchor, positive, negative, _ in loader:
            anchor, positive, negative = (
                anchor.to(device), 
                positive.to(device), 
                negative.to(device)
            )
            anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    # --- Configuration ---
    num_epochs = 1
    learning_rate = 1e-4
    triplet_margin = 0.5
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing on Device: {device} ---")

    # --- Model setup ---
    try:
        from modules import SiameseNetwork
        model = SiameseNetwork(embedding_dim=256).to(device)
    except ImportError:
        print("!! WARNING: SiameseNetwork not imported. Using dummy model !!")
        model = nn.Linear(1, 1).to(device)

    # --- Loss, Optimizer, Scheduler ---
    criterion = nn.TripletMarginLoss(margin=triplet_margin, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # --- Dataloaders ---
    train_loader, val_loader = get_data_loaders()

    # --- Training loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_triplet_model.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # --- Save best model checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_model_path)
            print(f"✅ Saved best model checkpoint at epoch {epoch} to '{best_model_path}'")

    # Optionally save full model (architecture + weights)
    full_model_path = os.path.join(save_dir, "final_triplet_model_full.pt")
    torch.save(model, full_model_path)
    print(f"\n✅ Training complete. Full model saved to '{full_model_path}'")


# --- Helper for inference ---
def load_model_for_inference(model_path, device=None):
    """
    Loads a saved SiameseNetwork for inference.
    Use with either the full model .pt or the state_dict .pth.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    if model_path.endswith(".pt"):
        # Full model (architecture + weights)
        model = torch.load(model_path, map_location=device)
        model.eval()
        print(f"✅ Loaded full model from '{model_path}'")
    else:
        # state_dict only, need to recreate model first
        from modules import SiameseNetwork
        model = SiameseNetwork(embedding_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Loaded model weights from '{model_path}' (epoch {checkpoint['epoch']})")

    return model


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(2000)
    main()
