"""
train.py

(ADD DESCRIPTION)

Author: Lalit Suresh
Date: 25th October 2025

"""

import os

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from tqdm import tqdm

from modules import SiameseNetwork, LesionClassifier
from dataset import get_data_loaders

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import EMBEDDING_DIM, LEARNING_RATE, EPOCHS, BATCH_SIZE, PARTIONED_IMGS_DIR

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_epoch(train_loader, device, model, classifier, optimizer, scaler, triplet_loss, class_loss):
    """
    Train siamese network and classifier for a single epoch
    """

    model.train()
    classifier.train()
    running_triplet_loss = 0.0
    running_class_loss = 0.0
    all_preds, all_labels = [], []


    pbar_train = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar_train):
        # Transfer to device (GPU/CPU)
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

            # Classification branch (binary → use sigmoid)
            logits = classifier(a_emb)
            class_loss_val = class_loss(logits.squeeze(), labels.float())

            total_loss = triplet_loss_val + class_loss_val

        # Backprop
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.sigmoid(logits).detach().cpu().round()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

        running_triplet_loss += triplet_loss_val.item()
        running_class_loss += class_loss_val.item()

        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nConfusion Matrix Training set:\n{cm}")

        # Calculate and display running accuracy
        running_acc = accuracy_score(all_labels, all_preds)
        pbar_train.set_postfix({'Triplet Loss': running_triplet_loss / (batch_idx + 1),
                                 'Class Loss': running_class_loss / (batch_idx + 1),
                                 'Acc': f'{running_acc:.4f}'})

    # Calculate final metrics for the epoch
    avg_triplet_loss = running_triplet_loss / len(train_loader)
    avg_class_loss = running_class_loss / len(train_loader)
    final_acc = accuracy_score(all_labels, all_preds)

    return avg_triplet_loss, avg_class_loss, final_acc



def validate_epoch(val_loader, device, model, classifier, triplet_loss, class_loss):
    """
    Validate siamese network and classifier for a single epoch
    """
    model.eval()
    classifier.eval()
    running_triplet_loss = 0.0
    running_class_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="Validation")
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar_val):
            # Transfer to device (GPU/CPU)
            anchor, positive, negative, labels = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                    labels.to(device),
            )
            a_emb, p_emb, n_emb = model(anchor, positive, negative)

            # --- Losses ---
            triplet_loss_val = triplet_loss(a_emb, p_emb, n_emb)
            logits = classifier(a_emb)
            class_loss_val = class_loss(logits.squeeze(), labels.float())

            total_loss = triplet_loss_val + class_loss_val

            preds = torch.sigmoid(logits).detach().cpu().round()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

            running_triplet_loss += triplet_loss_val.item()
            running_class_loss += class_loss_val.item()

            cm = confusion_matrix(all_labels, all_preds)

            print(f"\nConfusion Matrix Validation set:\n{cm}")

            running_acc = accuracy_score(all_labels, all_preds)
            pbar_val.set_postfix({'Triplet Loss': running_triplet_loss / (batch_idx + 1),
                                 'Class Loss': running_class_loss / (batch_idx + 1),
                                 'Acc': f'{running_acc:.4f}'})

    # Calculate final metrics for the epoch
    avg_triplet_loss = running_triplet_loss / len(val_loader)
    avg_class_loss = running_class_loss / len(val_loader)
    final_acc = accuracy_score(all_labels, all_preds)

    return avg_triplet_loss, avg_class_loss, final_acc

def main():
    """
    Performs training of the Siamese network + Classifier on the lesion dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Paths ===
    checkpoint_dir = "checkpoints"
    metrics_dir = "metrics"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    epoch_metrics_export_file = os.path.join("graphs", "epoch_metrics.csv")
    print("✅ Created / Checked necessary directories")

    # === Data loaders ===
    train_loader, val_loader = get_data_loaders()
    print("✅ Data loaders ready")

    # === Models, Losses, Optimizer, Scheduler ===
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = LesionClassifier(embedding_dim=EMBEDDING_DIM).to(device)

    triplet_loss = nn.TripletMarginLoss(margin=1.0).to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = GradScaler()

    # === Metrics storage ===
    train_triplet_losses, val_triplet_losses = [], []
    train_class_losses, val_class_losses = [], []
    train_accs, val_accs = [], []

    # === Checkpoint tracking ===
    best_val_acc = 0.0
    start_epoch = 0

    # === (Optional) Resume from checkpoint ===
    resume_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        classifier.load_state_dict(ckpt["classifier_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt["best_val_acc"]
        print(f"✅ Resumed from checkpoint at epoch {start_epoch} (best acc: {best_val_acc:.4f})")

    # === Training loop ===
    start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # --- Training ---
        train_triplet_loss, train_class_loss, train_acc = train_epoch(
            train_loader, device, model, classifier, optimizer, scaler, triplet_loss, classifier_loss
        )

        # --- Validation ---
        val_triplet_loss, val_class_loss, val_acc = validate_epoch(
            val_loader, device, model, classifier, triplet_loss, classifier_loss
        )

        # --- Record metrics ---
        train_triplet_losses.append(train_triplet_loss)
        val_triplet_losses.append(val_triplet_loss)
        train_class_losses.append(train_class_loss)
        val_class_losses.append(val_class_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train Triplet Loss: {train_triplet_loss:.4f}, "
            f"Train Class Loss: {train_class_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}\n"
            f"Val Triplet Loss: {val_triplet_loss:.4f}, "
            f"Val Class Loss: {val_class_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # --- Scheduler step (on validation accuracy) ---
        scheduler.step(val_acc)

        # --- Checkpointing ---
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "classifier_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        }

        # Save "last" checkpoint every epoch
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_checkpoint.pth"))

        # Save "best" checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_model_path)
            print(f"🏆 New best model saved at epoch {epoch+1} (val acc: {best_val_acc:.4f})")

        # Optional: periodic full checkpoint (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    df = pd.DataFrame({
        'epoch': range(len(EPOCHS)),
        'train_triplet_loss': train_triplet_losses,
        'val_triplet_loss': val_triplet_losses,
        'train_class_loss': train_class_losses,
        'val_class_loss': val_class_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })

    df.to_csv(epoch_metrics_export_file, index=False)

    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes.")
    print(f"✅ Best validation accuracy: {best_val_acc:.4f}")



if __name__ == "__main__":
    main()