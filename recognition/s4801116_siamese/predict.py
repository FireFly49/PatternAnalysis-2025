"""
predict.py

Generates predictoins using the Siamese 
Network using a trained model

Author: Lalit Suresh
Date: 25th October 2025

"""
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from dataset import get_data_loaders 
from modules import SiameseNetwork, LesionClassifier  
from config import EMBEDDING_DIM  


CHECKPOINT_PATH = "checkpoints/last_checkpoint(1).pth"
METRICS_CSV = "metrics/epoch_metrics.csv"
SAVE_DIR = "metrics"



def compute_confusion_matrix(model, device, classifier, loader, set_name):
    """
    Generates a confusion matrix using a given model
    on the validation and training sets.

    Args:
        device (torch.device): Device to run the computations on ('cuda' or 'cpu').
        model (nn.Module): Siamese network used to generate image embeddings.
        classifier (nn.Module): Classifier network that maps embeddings to class logits.
        loader (DataLoader): Data loader for confusion matrix.
        set_name (str): Name of confusion matrix.
    """
    all_preds, all_labels = [], []
    with torch.no_grad():
        for anchors, positives, negatives, labels in loader:
            anchors, labels = anchors.to(device), labels.to(device)
            embeddings = model.get_embeddings(anchors)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {set_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{set_name}.png"), dpi=200)
    plt.close()
    print(f"✅ Saved confusion matrix for {set_name} set.")
    return cm


def generate_tsne(model, device, loader, set_name):
    """
    Generates a t-sne plot using a given model
    on the validation and training sets.

    Args:
        model (nn.Module): Siamese network used to generate image embeddings.
        device (torch.device): Device to run the computations on ('cuda' or 'cpu').
        loader (DataLoader): Data loader for confusion matrix.
        set_name (str): Name of confusion matrix.

    """
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for anchors, positives, negatives, labels in loader:
            anchors, labels = anchors.to(device), labels.to(device)
            embeddings = model.get_embeddings(anchors)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=all_labels,
        palette="coolwarm", s=15, alpha=0.8
    )
    plt.title(f"t-SNE Visualization ({set_name} embeddings)")
    plt.legend(title="Label")
    plt.savefig(os.path.join(SAVE_DIR, f"tsne_{set_name}.png"), dpi=200)
    plt.close()
    print(f"✅ Saved t-SNE scatter plot for {set_name} set.")

def main():

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(METRICS_CSV)
    epochs = df["epoch"] if "epoch" in df.columns else range(len(df))

    plt.figure(figsize=(5, 3))
    plt.plot(epochs, df["train_triplet_loss"], label="Train Triplet Loss")
    plt.plot(epochs, df["val_triplet_loss"], label="Val Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title("Triplet Loss over Epochs")
    plt.xticks(np.arange(0, 21, step=2))
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "triplet_loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(epochs, df["train_class_loss"], label="Train Class Loss")
    plt.plot(epochs, df["val_class_loss"], label="Val Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title("Classification Loss over Epochs")
    plt.xticks(np.arange(0, 21, step=2))
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "class_loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(epochs, df["train_acc"], label="Train Accuracy")
    plt.plot(epochs, df["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xticks(np.arange(0, 21, step=2))
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"), dpi=200)
    plt.close()

    print("✅ Saved loss and accuracy plots.")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    classifier = LesionClassifier(embedding_dim=EMBEDDING_DIM).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    classifier.load_state_dict(checkpoint["classifier_state"])
    model.eval()
    classifier.eval()

    print("✅ Loaded trained model and classifier.")

    train_loader, val_loader = get_data_loaders()
    compute_confusion_matrix(model, device, classifier, train_loader, "train")
    compute_confusion_matrix(model, device, classifier, val_loader, "val")

    
    generate_tsne(model, device, train_loader, "train")
    generate_tsne(model, device, val_loader, "val")

    print("Plots saved in:", SAVE_DIR)

if __name__ == "__main__":
    mp.freeze_support()
    main()