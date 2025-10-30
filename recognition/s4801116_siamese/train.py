"""
train.py

(ADD DESCRIPTION)

Author: Lalit Suresh
Date: 25th October 2025

"""

import os

import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from modules import SiameseNetwork, LesionClassifier
from dataset import get_dataloaders

from sklearn.metrics import accuracy_score, f1_score

import time


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

        with autocast():
            a_emb, p_emb, n_emb = model(anchor, positive, negative)

            # --- Losses ---
            triplet_loss = triplet_loss(a_emb, p_emb, n_emb)

            # Classification branch (binary → use sigmoid)
            logits = classifier(a_emb)
            class_loss = class_loss(logits.squeeze(), labels.float())

            total_loss = triplet_loss + class_loss

        # Backprop
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = torch.sigmoid(logits).detach().cpu().round()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

        running_triplet_loss += triplet_loss.item()
        running_class_loss += triplet_loss.item()

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



def validate_epoch(val_loader, device, model, classifier):
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
            triplet_loss = triplet_loss(a_emb, p_emb, n_emb)
            logits = classifier(a_emb)
            class_loss = class_loss(logits.squeeze(), labels.float())

            total_loss = triplet_loss + class_loss

            preds = torch.sigmoid(logits).detach().cpu().round()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

            running_triplet_loss += triplet_loss.item()
            running_class_loss += triplet_loss.item()

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
    start = time.time()

    



    end = time.time()
    print(f"Training completed in {(end - start)/60:.2f} minutes.")


if __name__ == "__main__":
    main()