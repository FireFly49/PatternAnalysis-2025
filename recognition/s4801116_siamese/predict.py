"""
predict.py

Generates predictoins using the Siamese 
Network using a trained model

Author: Lalit Suresh
Date: 25th October 2025

"""
import torch
import torch.nn.functional as F
from modules import SiameseNetwork
from dataset import get_data_loaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

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
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        print(f"✅ Loaded full model from '{model_path}'")
    else:
        # state_dict only, need to recreate model first
        model = SiameseNetwork(embedding_dim=256).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Loaded model weights from '{model_path}' (epoch {checkpoint['epoch']})")

    return model

def evaluate_siamese_accuracy(model, data_loader, margin, device):
    """
    Evaluates the model's accuracy based on the Triplet Loss criterion:
    Distance(Anchor, Positive) < Distance(Anchor, Negative).
    
    The accuracy is the percentage of triplets that satisfy the condition 
    E_AP < E_AN (where E_AP is the distance between Anchor and Positive embeddings
    and E_AN is the distance between Anchor and Negative embeddings).
    
    Args:
        model (nn.Module): The trained Siamese network.
        data_loader (DataLoader): The validation or test DataLoader.
        margin (float): The margin used in the Triplet Loss.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The calculated Triplet Accuracy (0.0 to 1.0).
    """
    model.eval() # Set model to evaluation mode
    correct_triplets = 0
    total_triplets = 0
    
    with torch.no_grad():
        for anchor, positive, negative, _ in data_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Get embeddings
            output_A, output_P, output_N = model(anchor, positive, negative)
            
            # Calculate L2 (Euclidean) distance for Anchor-Positive and Anchor-Negative
            # E_AP: Distance between Anchor and Positive embeddings
            E_AP = F.pairwise_distance(output_A, output_P, p=2)
            # E_AN: Distance between Anchor and Negative embeddings
            E_AN = F.pairwise_distance(output_A, output_N, p=2)

            # A triplet is 'correct' if the Anchor is closer to the Positive than the Negative.
            correct_triplets += torch.sum(E_AP < E_AN).item()
            total_triplets += anchor.size(0)

    model.train() # Set model back to training mode
    return correct_triplets / total_triplets if total_triplets > 0 else 0.0

def get_verification_predictions(model, data_loader, distance_threshold, device):
    """
    Generates ground truth and predicted labels for a binary verification task
    based on the Siamese network's distances. This output is suitable for 
    calculating a Confusion Matrix using external libraries (e.g., scikit-learn).

    The task is: Given a pair, predict if they are the Same Class (1) or Different Class (0).
    A prediction is 'Same' (1) if Distance < threshold.

    Args:
        model (nn.Module): The trained Siamese network.
        data_loader (DataLoader): The validation or test DataLoader yielding (A, P, N).
        distance_threshold (float): The boundary distance (tau) for classification.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (y_true, y_pred), both are lists of 0s and 1s suitable for 
               calculating a confusion matrix or other metrics (e.g., AUC).
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for anchor, positive, negative, _ in data_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Get embeddings
            output_A, output_P, output_N = model(anchor, positive, negative)
            
            # 1. Anchor-Positive Pair (Ground Truth = Same Class/1)
            dist_AP = F.pairwise_distance(output_A, output_P, p=2)
            
            # Prediction: 1 (Same) if distance < threshold, 0 (Different) otherwise
            # Convert boolean comparison to int (0 or 1), then to CPU list
            pred_AP = (dist_AP < distance_threshold).int().cpu().tolist()
            
            y_pred.extend(pred_AP)
            y_true.extend([1] * len(pred_AP)) # True label is 1 (Same Class)

            # 2. Anchor-Negative Pair (Ground Truth = Different Classes/0)
            dist_AN = F.pairwise_distance(output_A, output_N, p=2)
            
            # Prediction: 1 (Same) if distance < threshold, 0 (Different) otherwise
            pred_AN = (dist_AN < distance_threshold).int().cpu().tolist()
            
            y_pred.extend(pred_AN)
            y_true.extend([0] * len(pred_AN)) # True label is 0 (Different Classes)

    model.train()
    return y_true, y_pred


def predict():
    pass


def main():
    predict()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = get_data_loaders()
    model = load_model_for_inference("checkpoints/final_triplet_model_full.pt")
    acc = evaluate_siamese_accuracy(model,data_loader=val_loader, margin=0.5, device=device)
    print(f"Triplet Accuracy on Validation Set: {acc*100:.2f}%")

    # --- 1. Define the Threshold ---
    # You need to experiment with this value. A good starting point might be 
    # your Triplet Margin, or the point that maximizes F1-score.
    DISTANCE_THRESHOLD = 0.8  # Example threshold (L2 distance)

    # --- 2. Generate True and Predicted Labels ---
    y_true, y_pred = get_verification_predictions(
        model=model, 
        data_loader=val_loader, 
        distance_threshold=DISTANCE_THRESHOLD, 
        device=device
    )

    # Convert lists to NumPy arrays for scikit-learn
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- 3. Calculate and Plot the Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Different (0)', 'Same (1)'])

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Verification Confusion Matrix (Threshold={DISTANCE_THRESHOLD})')
    plt.show()