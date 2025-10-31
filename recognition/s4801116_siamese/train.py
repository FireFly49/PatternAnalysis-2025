"""
train.py

(ADD DESCRIPTION)

Author: Lalit Suresh
Date: 25th October 2025

"""

import os
from collections import Counter

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from tqdm import tqdm

from modules import SiameseNetwork, LesionClassifier
from dataset import get_data_loaders

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from config import EMBEDDING_DIM, LEARNING_RATE, EPOCHS

import time

import pandas as pd


def train_epoch(train_loader, device, model, classifier, optimizer, scaler, triplet_loss, class_loss):
    """
    Performs training on the siamese network and classifier for one epoch.

    Args: 
        train_loader (DataLoader): DataLoader providing batches of triplets (anchor, positive, negative) along with class labels.
        device (torch.device): Device to run the computations on ('cuda' or 'cpu').
        model (nn.Module): Siamese network used to generate image embeddings.
        classifier (nn.Module): Classifier network that maps embeddings to class logits.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scaler (torch.cuda.amp.GradScaler): Mixed precision gradient scaler for stable training.
        triplet_loss (nn.Module): Loss function for embedding distance learning.
        class_loss (nn.Module): Loss function for classification (e.g., CrossEntropyLoss).
    
    Returns:
        avg_triplet_loss (float): Mean triplet loss over all training batches.
        avg_class_loss (float): Mean classification loss over all training batches.
        final_acc (float): Average classification accuracy across the epoch.
    """

    model.train()
    classifier.train()
    running_triplet_loss = 0.0
    running_class_loss = 0.0
    all_preds, all_labels = [], []


    pbar_train = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar_train):

        anchor, positive, negative, labels = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
                labels.to(device),
        )

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            a_emb, p_emb, n_emb = model(anchor, positive, negative)

            # --- Losses ---
            triplet_loss_val = triplet_loss(a_emb, p_emb, n_emb)
            logits = classifier(a_emb)    # shape: [batch_size, 2]
            labels = labels.long()        # ensure integer labels
            class_loss_val = class_loss(logits, labels)

            total_loss = triplet_loss_val + class_loss_val

        # Backprop
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(logits, dim=1)  # gives 0 or 1
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        running_triplet_loss += triplet_loss_val.item()
        running_class_loss += class_loss_val.item()

        # Calculate and display running accuracy
        running_acc = balanced_accuracy_score(all_labels, all_preds)
        pbar_train.set_postfix({'Triplet Loss': running_triplet_loss / (batch_idx + 1),
                                 'Class Loss': running_class_loss / (batch_idx + 1),
                                 'Acc': f'{running_acc:.4f}'})
        
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix Training set:\n{cm}")

    # Calculate final metrics for the epoch
    avg_triplet_loss = running_triplet_loss / len(train_loader)
    avg_class_loss = running_class_loss / len(train_loader)
    final_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_triplet_loss, avg_class_loss, final_acc



def validate_epoch(val_loader, device, model, classifier, triplet_loss, class_loss):
    """
    Validate siamese network and classifier for a single epoch.

    Args
        val_loader: (DataLoader): DataLoader for training set.
        device (torch.device): Device to run the computations on ('cuda' or 'cpu').
        model (nn.Module): Siamese network used to generate image embeddings.
        classifier (nn.Module): Classifier network that maps embeddings to class logits.
        triplet_loss (nn.Module): Loss function for embedding distance learning.
        class_loss (nn.Module): Loss function for classification (e.g., CrossEntropyLoss).

    Returns:
        avg_triplet_loss (float): Mean triplet loss over all training batches.
        avg_class_loss (float): Mean classification loss over all training batches.
        final_acc (float): Average classification accuracy across the epoch.
    """
    model.eval()
    classifier.eval()
    running_triplet_loss = 0.0
    running_class_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="Validation")
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar_val):

            anchor, positive, negative, labels = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                    labels.to(device),
            )
            a_emb, p_emb, n_emb = model(anchor, positive, negative)

            triplet_loss_val = triplet_loss(a_emb, p_emb, n_emb)
            logits = classifier(a_emb)       # shape: [batch_size, 2]
            labels = labels.long()           # ensure integer labels
            class_loss_val = class_loss(logits, labels)

            total_loss = triplet_loss_val + class_loss_val

            preds = torch.argmax(logits, dim=1)  # gives 0 or 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_triplet_loss += triplet_loss_val.item()
            running_class_loss += class_loss_val.item()

            running_acc = balanced_accuracy_score(all_labels, all_preds)
            pbar_val.set_postfix({'Triplet Loss': running_triplet_loss / (batch_idx + 1),
                                 'Class Loss': running_class_loss / (batch_idx + 1),
                                 'Acc': f'{running_acc:.4f}'})
            
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix Validation set:\n{cm}")

    # Calculate final metrics for the epoch
    avg_triplet_loss = running_triplet_loss / len(val_loader)
    avg_class_loss = running_class_loss / len(val_loader)

    final_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_triplet_loss, avg_class_loss, final_acc


def main():
    """
    Performs training of the Siamese network + Classifier on the lesion dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = "checkpoints"
    metrics_dir = "metrics"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    epoch_metrics_export_file = os.path.join(metrics_dir, "epoch_metrics.csv")
    print("✅ Created / Checked necessary directories")

    train_loader, val_loader = get_data_loaders()
    print("✅ Data loaders ready")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = LesionClassifier(embedding_dim=EMBEDDING_DIM).to(device)

    train_labels = train_loader.dataset.all_labels
    class_counts = Counter(train_labels)
    benign_count = class_counts[0]
    malignant_count = class_counts[1]

    ratio = benign_count / malignant_count
    class_weights = torch.tensor([1.0, min(ratio, 5.0)], device=device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0).to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights).to(device)

    optimizer = Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = GradScaler()

    train_triplet_losses, val_triplet_losses = [], []
    train_class_losses, val_class_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    start_epoch = 0

    resume_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        classifier.load_state_dict(ckpt["classifier_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"✅ Resumed from checkpoint at epoch {start_epoch} (best acc: {best_val_acc:.4f})")

    start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Backbone freeze/unfreeze logic
        if epoch < 3:
            # Freeze the backbone for first 3 epochs
            for param in model.backbone.parameters():
                param.requires_grad = False
            if epoch == 0:
                print("Backbone frozen for initial warmup phase")
        else:
            # Unfreeze the backbone for fine-tuning
            for param in model.backbone.parameters():
                param.requires_grad = True
            if epoch == 3:
                print("Backbone unfrozen for fine-tuning")

        train_triplet_loss, train_class_loss, train_acc = train_epoch(
            train_loader, device, model, classifier, optimizer, scaler, triplet_loss, classifier_loss
        )

        val_triplet_loss, val_class_loss, val_acc = validate_epoch(
            val_loader, device, model, classifier, triplet_loss, classifier_loss
        )

        train_triplet_losses.append(train_triplet_loss)
        val_triplet_losses.append(val_triplet_loss)
        train_class_losses.append(train_class_loss)
        val_class_losses.append(val_class_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train Triplet Loss: {train_triplet_loss:.4f}, "
            f"Train Class Loss: {train_class_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Triplet Loss: {val_triplet_loss:.4f}, "
            f"Val Class Loss: {val_class_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
        )

        scheduler.step(val_acc)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "classifier_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_model_path)
            print(f"🏆 New best model saved at epoch {epoch+1} (val acc: {best_val_acc:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    df = pd.DataFrame({
        'epoch': range(1, len(train_accs) + 1),
        'train_triplet_loss': train_triplet_losses,
        'val_triplet_loss': val_triplet_losses,
        'train_class_loss': train_class_losses,
        'val_class_loss': val_class_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
    })
    df.to_csv(epoch_metrics_export_file, index=False)

    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes.")
    print(f"✅ Best validation acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()